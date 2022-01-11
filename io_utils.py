import numpy as np
import os
import glob
import argparse
import models.backbone as backbone
import models.resnet as resnet
import models.googlenet as googlenet
import torchvision
import models.vgg as vgg
#nohup python train.py --model ResNet18 --method relationnet_softmax --train_aug>resnet18.out
#CUDA_VISIBLE_DEVICES=1 nohup python train.py --model ResNet50 --method relationnet_softmax --train_aug>resnet50.out

#swin
from models.swin.config_swin import get_config, get_config_cifar
from models.swin.models_swin import build_model
from functools import partial
cfg_file = './models/swin/configs_swin/swin_tiny_patch4_window7_224.yaml'
config_swin = get_config_cifar(cfg_file)
swin_m = partial(build_model, config_swin)

from models.ceit import ceit_model
from timm.models import create_model
#from functools import partial
ceit_m = partial(create_model,'ceit_tiny_patch16_224', 
                  pretrained=False,
                  drop_rate=0.0,
                  drop_path_rate=0.1,
                  drop_block_rate=None,
                  leff_local_size=3,
                  leff_with_bn=True)


model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            Conv4224 = backbone.Conv4224,
            Conv6224 = backbone.Conv6224,
            ResNet10 = backbone.ResNet10,  
            ResNet18 = backbone.ResNet18,  #
            ResNet34 = backbone.ResNet34,  
            ResNet50 = backbone.ResNet50,  # 
            ResNet101 = backbone.ResNet101,
            visionresnet18 = resnet.resnet18,
            visiongoogle = googlenet.googlenet,
            visionvgg = vgg.vgg19,
            visionswin = swin_m,
            visionceit = ceit_m) 

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char/cifar10')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101} /visionresnet18/visiongoogle/visionvgg/visionswin/visionceit') # 50 and 101 are not used in the paper
    parser.add_argument('--model_name', type=str, default = '')
    parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--tweights'    , default='./Save/checkpoints/CUB/Conv6_protonet_aug_5way_1shot/best_model.tar') # 如果蒸馏，teacher weights 保存地址；否则为预训练模型保存地址
    parser.add_argument('--savename'  , type=str, default="scratch_amazon", help='further adaptation in test time or not')
    parser.add_argument('--novel_file', type=str, default='./Save/features/{}/{}/visionresnet18_protonet_aug_5way_1shot/val.hdf5')
    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
        
        # only use in distill ( 这个框架下无法蒸馏 )
        parser.add_argument('--distill'     , default='no',type=str,help='no/kd') # much more haven't added-crd/attention/similarity...... in distiller_zoo
        parser.add_argument('--smodel'      , default='Conv4',      help='student model:  Conv{4|6} / ResNet{10|18|34|50|101}')
        parser.add_argument('--tmodel'      , default='Conv6',      help='teacher model:  Conv{4|6} / ResNet{10|18|34|50|101}')
        parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')  # KL distillation    

        # mask prune 
        parser.add_argument('--mask_prune' , action='store_true', help = ''' 使用mask进行prune，训练模型 ''') #
        parser.add_argument('--mask_weights', default = 'mask.tar', help = ''' 基于topk保存的每个模型 ''') 
        parser.add_argument('--mask_ratio', type=str, default = '0.8')


    elif script == 'save_features':
        parser.add_argument('--split'       , default='val', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='val', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        
    elif script == 'extract':
        parser.add_argument('--weights_pth', type=str, default = './Save/checkpoints/CUB/sConv4_tConv6_protonet_aug_5way_1shot_distill_depara4_kd/best_model.tar') # resnet18 for protonet './Save/checkpoints/CUB/ResNet18_protonet_aug_5way_1shot/best_model.tar'
        parser.add_argument('--save_dir'   , type=str, default = './depara_extracts/conv4_kd_depara4_protonet')
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
