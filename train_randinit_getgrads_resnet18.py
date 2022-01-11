import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

# import sys
# sys.path.append("./")

import configs
import models.backbone as backbone
import models.resnet as resnet
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file  
import pdb

# torchvision resnet18
#CUDA_VISIBLE_DEVICES=3 nohup python train_randinit_getgrads_resnet18.py --method protonet --n_shot 1  --model_name randinit_resnet18 --train_aug>protonet_shot1_randinitresnet18_randinit.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_randinit_getgrads_resnet18.py --method protonet --n_shot 1  --model_name randinit_resnet18_miniimagenet --dataset miniImagenet --train_aug>protonet_shot1_randinitresnet18_miniImagenet_randinit.out


def train_randinit(params):   
    # only support protonet
    if params.method == 'protonet':
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params    = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor   

        #model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        model = ProtoNet(resnet.resnet18, **train_few_shot_params)
        model = model.cuda()
        model.zero_grad()
        #criterion
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Only support protonet!')

    start_epoch = params.start_epoch
    stop_epoch  = params.stop_epoch

    count = 0
    for epoch in range(start_epoch,stop_epoch+1):
        #train
        print_freq = 20
        avg_loss=0
        for i, (x,_ ) in enumerate(base_loader):
            new_model = None           
            if params.method == 'protonet':
                new_model = ProtoNet(resnet.resnet18, **train_few_shot_params)
                new_model = new_model.cuda()

            new_model.n_query = x.size(1) - new_model.n_support   # x.size- [5,21,3,84,84]    
            if new_model.change_way:
                new_model.n_way  = x.size(0)
            
            new_model.zero_grad()
            scores = new_model.set_forward(x)

            y_query = torch.from_numpy(np.repeat(range(new_model.n_way ), new_model.n_query ))
            y_query = Variable(y_query.cuda())
            loss    = criterion(scores, y_query)

            loss.backward()

            # copy grads from new model to model
            for p,new_p in zip(model.parameters(),new_model.parameters()):
                p.data = (p.data*count + new_p.grad)/(count+1)
            # model.feature.trunk[0].weight.data == new_model.feature.trunk[0].weight.grad.abs()
            # (model.feature.trunk[0].weight.data != new_model.feature.trunk[0].weight.grad.abs()).sum() -> 0
            count += 1

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Count {:d}'.format(epoch, i, len(base_loader), count))
       
        if (epoch % params.save_freq==0) or (epoch==stop_epoch):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')


    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json' 
         
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default
     
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model_name, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    train_randinit(params)
