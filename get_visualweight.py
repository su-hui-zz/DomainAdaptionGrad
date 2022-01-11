'''
计算两个相同结构的网络，训练得到的grad的余弦距离
'''
import torch
import argparse
import matplotlib.pyplot as plt
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file 
import resnet
import os
import cv2
import numpy as np
import pdb

def m_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model'  , default='visionresnet18',  help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method' , default='protonet',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--weights1', type=str, default = './Save/checkpoints/CUB/visionresnet18_protonet_aug_5way_1shot/600.tar') # resnet18 CUB for protonet 
    parser.add_argument('--weights2', type=str, default = './Save/checkpoints/miniImagenet/visionresnet18_protonet_aug_5way_1shot/600.tar') # resnet18 CUB for protonet 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--n_shot'      , default=1, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--save_dir', type=str, default = './Save/grad/CUB_miniImage_resnet18/')
    args = parser.parse_args()
    return args

# weights 可视化
def get_visual(model1, model2, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for (name1, module1), (name2, module2) in zip(model1.named_modules(),model2.named_modules()):
        if type(module1) == torch.nn.Conv2d and type(module2) == torch.nn.Conv2d:
            assert name1 == name2
            n1,n2,kw,kh = module1.weight.size()
            w1 = module1.weight.permute(0,2,1,3)
            w1 = w1.reshape(n1*kw,n2*kh)
            w1 = (w1-w1.min()) / (w1.max()-w1.min()) * 255.

            w2 = module2.weight.permute(0,2,1,3)
            w2 = w2.reshape(n1*kw,n2*kh)
            w2 = (w2-w2.min()) / (w2.max() - w2.min()) * 255.
            
            medium = torch.zeros(n1*kw, n2*kh//2)

            w = torch.cat([w1,medium, w2],dim=-1)

            save_path = os.path.join(save_dir, '.'.join(name1.split('.')[1:])+'.png')
            cv2.imwrite(save_path, w.detach().numpy())


# 获取矩阵协方差信息
def get_covinf(model):
    for name, module in model.named_modules():
        if type(module) == torch.nn.Conv2d:
            n1,n2,kw,kh = module.weight.size()
            w = module.weight.reshape(n1*n2,kw*kh)
            cor = np.corrcoef(w.detach().numpy())
            print(name)
            print(cor)
            print('\n')

def get_mean_kernel(model):
    f = open('kernel_mean.txt','w')
    for name, module in model.named_modules():
        if type(module) == torch.nn.Conv2d:
            n1,n2,kw,kh = module.weight.size()
            w = module.weight.reshape(n1*n2,kw*kh)
            mv = w.mean(dim=0).detach().numpy().tolist()
            f.write(name+'\n')
            strv = [str(round(v,2)) for v in mv]
            strv = ', '.join(strv)
            f.write(strv)
            f.write('\n\n')

if __name__ == '__main__':
    args = m_parse()

    model = None
    if args.method == 'protonet':
        train_few_shot_params   = dict(n_way = args.train_n_way, n_support = args.n_shot) 
        model1 = ProtoNet(model_dict[args.model], **train_few_shot_params)
        model2 = ProtoNet(model_dict[args.model], **train_few_shot_params)
    else:
        raise ValueError('Only support protonet!')
    
    model1.load_state_dict(torch.load(args.weights1)['state'])
    model2.load_state_dict(torch.load(args.weights2)['state'])

    #get_visual(model1, model2, args.save_dir)

    #get_covinf(model1)
    get_mean_kernel(model1)



