import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import models.backbone as backbone

from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file  
import pdb

##################################################################
# swin
# 加载cifar10/cifar100预训练模型，在cub数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionswin --method protonet --n_shot 1 --dataset CUB --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_swintiny_pretrained_sz224/checkpoints/swintiny_pre-epoch120-acc78.420.pth --model_name cifar100_sz224_swintiny_gradcub --train_aug>protonet_shot1_cifar100_swintiny_gradcub.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionswin --method protonet --n_shot 1 --dataset CUB --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_swintiny_pretrained_sz224/checkpoints/swintiny_pre-epoch120-acc92.790.pth --model_name cifar10_sz224_swintiny_gradcub --train_aug>protonet_shot1_cifar10_swintiny_gradcub.out

# 加载cifar10/cifar100预训练模型，在cifar10数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionswin --method protonet --n_shot 1 --dataset cifar10 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_swintiny_pretrained_sz224/checkpoints/swintiny_pre-epoch120-acc78.420.pth --model_name cifar100_sz224_swintiny_gradcifar10 --train_aug>protonet_shot1_cifar100_swintiny_gradcifar10.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionswin --method protonet --n_shot 1 --dataset cifar10 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_swintiny_pretrained_sz224/checkpoints/swintiny_pre-epoch120-acc92.790.pth --model_name cifar10_sz224_swintiny_gradcifar10 --train_aug>protonet_shot1_cifar10_swintiny_gradcifar10.out

# 加载cifar10/cifar100预训练模型，在cifar100数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionswin --method protonet --n_shot 1 --dataset cifar100 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_swintiny_pretrained_sz224/checkpoints/swintiny_pre-epoch120-acc78.420.pth --model_name cifar100_sz224_swintiny_gradcifar100 --train_aug>protonet_shot1_cifar100_swintiny_gradcifar100.out
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visionswin --method protonet --n_shot 1 --dataset cifar100 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_swintiny_pretrained_sz224/checkpoints/swintiny_pre-epoch120-acc92.790.pth --model_name cifar10_sz224_swintiny_gradcifar100 --train_aug>protonet_shot1_cifar10_swintiny_gradcifar100.out

##################################################################
# ceit
# 加载cifar10/cifar100预训练模型，在cub数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=? nohup python train_preinit_getgrads.py --model visionceit --method protonet --n_shot 1 --dataset CUB --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_ceit_pretrained_sz224/checkpoints/best.pth --model_name cifar100_sz224_ceit_gradcub --train_aug>protonet_shot1_cifar100_ceit_gradcub.out
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visionceit --method protonet --n_shot 1 --dataset CUB --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_ceit_pretrained_sz224/checkpoints/ceit_tiny-epoch120-acc85.980.pth --model_name cifar10_sz224_ceit_gradcub --train_aug>protonet_shot1_cifar10_ceit_gradcub.out

# 加载cifar10/cifar100预训练模型，在cifar10数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=? nohup python train_preinit_getgrads.py --model visionceit --method protonet --n_shot 1 --dataset cifar10 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_ceit_pretrained_sz224/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz224_ceit_gradcifar10 --train_aug>protonet_shot1_cifar100_ceit_gradcifar10.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionceit --method protonet --n_shot 1 --dataset cifar10 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_ceit_pretrained_sz224/checkpoints/ceit_tiny-epoch120-acc85.980.pth --model_name cifar10_sz224_ceit_gradcifar10 --train_aug>protonet_shot1_cifar10_ceit_gradcifar10.out

# 加载cifar10/cifar100预训练模型，在cifar100数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=? nohup python train_preinit_getgrads.py --model visionceit --method protonet --n_shot 1 --dataset cifar100 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_ceit_pretrained_sz224/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz84_vgg_gradcifar100 --train_aug>protonet_shot1_cifar100_vgg_gradcifar100.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionceit --method protonet --n_shot 1 --dataset cifar100 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_ceit_pretrained_sz224/checkpoints/ceit_tiny-epoch120-acc85.980.pth --model_name cifar10_sz224_ceit_grad100 --train_aug>protonet_shot1_cifar10_ceit_gradcifar100.out

##################################################################
# torchvision vgg
# 加载cifar10/cifar100预训练模型，在cub数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset CUB --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz84_vgg_gradcub --train_aug>protonet_shot1_cifar100_vgg_gradcub.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset CUB --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch120-acc90.430.pth --model_name cifar10_sz84_vgg_gradcub --train_aug>protonet_shot1_cifar10_vgg_gradcub.out

# 加载cifar10/cifar100预训练模型，在cifar10数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset cifar10 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz84_vgg_gradcifar10 --train_aug>protonet_shot1_cifar100_vgg_gradcifar10.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset cifar10 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch120-acc90.430.pth --model_name cifar10_sz84_vgg_gradcifar10 --train_aug>protonet_shot1_cifar10_vgg_gradcifar10.out

# 加载cifar10/cifar100预训练模型，在cifar100数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset cifar100 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz84_vgg_gradcifar100 --train_aug>protonet_shot1_cifar100_vgg_gradcifar100.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset cifar100 --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch120-acc90.430.pth --model_name cifar10_sz84_vgg_gradcifar100 --train_aug>protonet_shot1_cifar10_vgg_gradcifar100.out

# 加载cifar10/cifar100的预训练模型,在mini-imagenet数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset miniImagenet --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz84_vgg_gradminiimagenet --train_aug>protonet_shot1_cifar100_vgg_gradminiimagenet.out
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset miniImagenet --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch120-acc90.430.pth --model_name cifar10_sz84_vgg_gradminiimagenet --train_aug>protonet_shot1_cifar10_vgg_gradminiimagenet.out

# 加载cifar10/cifar100的预训练模型,在omnglot数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset omniglot --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch200-acc71.820.pth --model_name cifar100_sz84_vgg_gradomniglot --train_aug>protonet_shot1_cifar100_vgg_gradomniglot.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionvgg --method protonet --n_shot 1 --dataset omniglot --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_vgg_pretrained_sz84/checkpoints/vgg19_vision_pre-epoch120-acc90.430.pth --model_name cifar10_sz84_vgg_gradomniglot --train_aug>protonet_shot1_cifar10_vgg_gradomniglot.out


##################################################################
# resnet18
# domainnet 数据集（imagenet pre)
# 分别加载5个域的预训练模型，在clipart数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_clipart --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/clipart/checkpoints/resnet18_vision-epoch130-acc63.483.pth --model_name clipart_resnet18_gradclipart --train_aug>protonet_shot1_clipart_resnet18_gradclipart.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_painting --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/clipart/checkpoints/resnet18_vision-epoch130-acc63.483.pth --model_name clipart_resnet18_gradpainting --train_aug>protonet_shot1_clipart_resnet18_gradpainting.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_real --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/clipart/checkpoints/resnet18_vision-epoch130-acc63.483.pth --model_name clipart_resnet18_real --train_aug>protonet_shot1_clipart_resnet18_gradreal.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_quickdraw --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/clipart/checkpoints/resnet18_vision-epoch130-acc63.483.pth --model_name clipart_resnet18_gradquickdraw  --train_aug>protonet_shot1_clipart_resnet18_gradquickdraw.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_sketch --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/clipart/checkpoints/resnet18_vision-epoch130-acc63.483.pth --model_name clipart_resnet18_gradsketch  --train_aug>protonet_shot1_clipart_resnet18_gradsketch.out

# 分别加载5个域的预训练模型，在painting数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_clipart --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/painting/checkpoints/resnet18_vision-epoch130-acc48.517.pth --model_name painting_resnet18_gradclipart --train_aug>protonet_shot1_painting_resnet18_gradclipart.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_painting --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/painting/checkpoints/resnet18_vision-epoch130-acc48.517.pth --model_name painting_resnet18_gradpainting --train_aug>protonet_shot1_painting_resnet18_gradpainting.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_real --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/painting/checkpoints/resnet18_vision-epoch130-acc48.517.pth --model_name painting_resnet18_real --train_aug>protonet_shot1_painting_resnet18_gradreal.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_quickdraw --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/painting/checkpoints/resnet18_vision-epoch130-acc48.517.pth --model_name painting_resnet18_gradquickdraw  --train_aug>protonet_shot1_painting_resnet18_gradquickdraw.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_sketch --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/painting/checkpoints/resnet18_vision-epoch130-acc48.517.pth --model_name painting_resnet18_gradsketch  --train_aug>protonet_shot1_painting_resnet18_gradsketch.out

# 分别加载5个域的预训练模型，在real数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_clipart --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/real/checkpoints/resnet18_vision-epoch130-acc66.965.pth --model_name real_resnet18_gradclipart --train_aug>protonet_shot1_real_resnet18_gradclipart.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_painting --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/real/checkpoints/resnet18_vision-epoch130-acc66.965.pth --model_name real_resnet18_gradpainting --train_aug>protonet_shot1_real_resnet18_gradpainting.out
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_real --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/real/checkpoints/resnet18_vision-epoch130-acc66.965.pth --model_name real_resnet18_real --train_aug>protonet_shot1_real_resnet18_gradreal.out
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_quickdraw --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/real/checkpoints/resnet18_vision-epoch130-acc66.965.pth --model_name real_resnet18_gradquickdraw  --train_aug>protonet_shot1_real_resnet18_gradquickdraw.out
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_sketch --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/real/checkpoints/resnet18_vision-epoch130-acc66.965.pth --model_name real_resnet18_gradsketch  --train_aug>protonet_shot1_real_resnet18_gradsketch.out

# 分别加载5个域的预训练模型，在quickdraw数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_clipart --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/quickdraw/checkpoints/resnet18_vision-epoch130-acc68.286.pth --model_name quickdraw_resnet18_gradclipart --train_aug>protonet_shot1_quickdraw_resnet18_gradclipart.out
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_painting --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/quickdraw/checkpoints/resnet18_vision-epoch130-acc68.286.pth --model_name quickdraw_resnet18_gradpainting --train_aug>protonet_shot1_quickdraw_resnet18_gradpainting.out
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_real --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/quickdraw/checkpoints/resnet18_vision-epoch130-acc68.286.pth --model_name quickdraw_resnet18_real --train_aug>protonet_shot1_quickdraw_resnet18_gradreal.out
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_quickdraw --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/quickdraw/checkpoints/resnet18_vision-epoch130-acc68.286.pth --model_name quickdraw_resnet18_gradquickdraw  --train_aug>protonet_shot1_quickdraw_resnet18_gradquickdraw.out
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_sketch --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/quickdraw/checkpoints/resnet18_vision-epoch130-acc68.286.pth --model_name quickdraw_resnet18_gradsketch  --train_aug>protonet_shot1_quickdraw_resnet18_gradsketch.out

# 分别加载5个域的预训练模型，在sketch数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_clipart --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/sketch/checkpoints/resnet18_vision-epoch130-acc52.572.pth --model_name sketch_resnet18_gradclipart --train_aug>protonet_shot1_sketch_resnet18_gradclipart.out
#CUDA_VISIBLE_DEVICES=2 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_painting --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/sketch/checkpoints/resnet18_vision-epoch130-acc52.572.pth --model_name sketch_resnet18_gradpainting --train_aug>protonet_shot1_sketch_resnet18_gradpainting.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_real --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/sketch/checkpoints/resnet18_vision-epoch130-acc52.572.pth --model_name sketch_resnet18_real --train_aug>protonet_shot1_sketch_resnet18_gradreal.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_quickdraw --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/sketch/checkpoints/resnet18_vision-epoch130-acc52.572.pth --model_name sketch_resnet18_gradquickdraw  --train_aug>protonet_shot1_sketch_resnet18_gradquickdraw.out
#CUDA_VISIBLE_DEVICES=3 nohup python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_sketch --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/domainnet_resnet18_pretrained/sketch/checkpoints/resnet18_vision-epoch130-acc52.572.pth --model_name sketch_resnet18_gradsketch  --train_aug>protonet_shot1_sketch_resnet18_gradsketch.out


##################################################################
# google
# cifar10/cifar100的预训练模型在mini-imagenet数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=0 nohup python train_preinit_getgrads.py --model visiongoogle --method protonet --n_shot 1 --dataset miniImagenet --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_google_pretrained_sz84/checkpoints/googlenet_vision_pre-epoch200-acc74.020.pth --model_name cifar100_sz84_google_gradminiimagenet --train_aug>protonet_shot1_cifar100_google_gradminiimagenet.out
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visiongoogle --method protonet --n_shot 1 --dataset miniImagenet --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_google_pretrained_sz84/checkpoints/googlenet_vision_pre-epoch120-acc95.050.pth --model_name cifar10_sz84_google_gradminiimagenet --train_aug>protonet_shot1_cifar10_google_gradminiimagenet.out

# cifar10/cifar100的预训练模型在omnglot数据集上获取梯度信息
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visiongoogle --method protonet --n_shot 1 --dataset omniglot --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar100_google_pretrained_sz84/checkpoints/googlenet_vision_pre-epoch200-acc74.020.pth --model_name cifar100_sz84_google_gradomniglot --train_aug>protonet_shot1_cifar100_google_gradomniglot.out
#CUDA_VISIBLE_DEVICES=1 nohup python train_preinit_getgrads.py --model visiongoogle --method protonet --n_shot 1 --dataset omniglot --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/cifar10_google_pretrained_sz84/checkpoints/googlenet_vision_pre-epoch120-acc95.050.pth --model_name cifar10_sz84_google_gradomniglot --train_aug>protonet_shot1_cifar10_google_gradomniglot.out


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

        model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        state_dict = {}
        if 'epoch' in params.tweights:
            weights = torch.load(params.tweights)#['model_state_dict']
        else:
            weights = torch.load(params.tweights)
        for k,v in weights.items():
            if 'head.weight' not in k and 'head.bias' not in k and 'classifier' not in k and 'fc' not in k and 'aux1' not in k and 'aux2' not in k:
                state_dict[k] = v
        
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
                new_model = ProtoNet(model_dict[params.model], **train_few_shot_params)
                new_model.feature.load_state_dict(state_dict, strict=True) 
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
    params.cfg_file = './models/swin/configs_swin/swin_tiny_patch4_window7_224.yaml'


    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json' 
         
    if 'Conv' in params.model: #vgg/google/resnet18
        #if params.dataset in ['omniglot', 'cross_char']:
        #    image_size = 28
        #else:
        #    image_size = 84
        image_size = 84  # vgg/google/resnet18
    else:
        image_size = 224 # ceit

    if params.dataset in ['omniglot', 'cross_char']:
        #assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        if params.model == 'Conv4' and not params.train_aug:
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
