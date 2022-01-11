import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import models.backbone as backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML

from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
import pdb
#python ./test_office_resnet18.py --dataset oc_amazon --savename scratch_amazon --model visionresnet18 --method protonet --train_aug  --n_shot 1 
#python ./test_office_resnet18.py --dataset oc_amazon --savename scratch_caltech10 --model visionresnet18 --method protonet --train_aug  --n_shot 1 
#python ./test_office_resnet18.py --dataset oc_amazon --savename scratch_dslr --model visionresnet18 --method protonet --train_aug  --n_shot 1 
#python ./test_office_resnet18.py --dataset oc_amazon --savename scratch_webcam --model visionresnet18 --method protonet --train_aug  --n_shot 1 

#5way1shot
# model-scratch amazon
#oc_amazon    - 59.81% +- 1.00%
#oc_caltech10 - 31.27% +- 0.64%
#oc_dslr - 56.46% +- 0.85%
#oc_webcam - 52.86% +- 0.75%


def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        img_num  = len(perm_ids)
        z_all.append( [ np.squeeze( img_feat[perm_ids[i%img_num]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc

if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6': 
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S': 
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    # state_dict = {}
    # weights = torch.load(params.tweights)#['model_state_dict']
    # for k,v in weights.items():
    #     if 'fc' not in k:
    #         state_dict[k] = v
    # model.feature.load_state_dict(state_dict, strict=True)#['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    
    #novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
    novel_file = params.novel_file.format(params.savename, params.dataset)

    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
        acc_all.append(acc)

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print(params.savename, '-', params.dataset)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

