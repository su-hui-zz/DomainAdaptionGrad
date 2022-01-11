'''
根据grad值取前topk,构建mask(>threshold,1; <threshold,0),用于后续模型剪枝训练。
'''
import numpy as np
import torch
from torch import autograd
import matplotlib.pyplot as plt
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file 
import argparse
import os
import pdb

def m_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model'  , default='visionresnet18',  help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='protonet',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    #parser.add_argument('--weights', type=str, default = './Save/checkpoints/CUB/ResNet18_protonet_aug_5way_1shot/600.tar') # resnet18 for protonet './Save/checkpoints/omniglot/ResNet18_protonet_aug_5way_1shot/200.tar'
    parser.add_argument('--weights', type=str, default = './Save/checkpoints/CUB/visionresnet18_protonet_aug_5way_1shot/450.tar') # resnet18 for protonet './Save/checkpoints/omniglot/ResNet18_protonet_aug_5way_1shot/200.tar'
    parser.add_argument('--save_m',  type=str, default = 'mask_{}.tar')
    parser.add_argument('--mratio',  type=list,default = [0.4]*9, help = '80% weights remains')
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--n_shot'      , default=1, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    args = parser.parse_args()
    return args


# def rank_mask(model, m_mask, mratio, weight_pth, save_m):
#     m_ind = -1
#     last_ind = -1
#     for (name, module),(m_name, m_module) in zip(model.named_modules(), m_mask.named_modules()):
#         if type(module) == torch.nn.Conv2d:
#             assert name == m_name
#             pdb.set_trace()
#             # 保证每个block的多个conv 的 ratio一致
#             if last_ind != int(name.split('.')[2]): #feature.trunk.0
#                 last_ind = int(name.split('.')[2])
#                 m_ind += 1
            
#             n1,n2,kw,kh = module.weight.size()
#             w,idx = module.weight.flatten().sort(descending=True)
            
#             j = int(mratio[m_ind]*w.numel())
#             flat_out = m_module.weight.flatten()
#             flat_out[idx[j:]] = 0
#             flat_out[idx[:j]] = 1
#             flat_out.reshape(n1,n2,kw,kh)
    
#     out_dir = os.path.dirname(weight_pth)
#     outfile = os.path.join(out_dir, save_m.format(mratio[-1]))
#     torch.save(m_mask.state_dict(), outfile)
    
    
def rank_mask(model, m_mask, mratio, weight_pth, save_m):
    for (name, module),(m_name, m_module) in zip(model.named_modules(), m_mask.named_modules()):
        if type(module) == torch.nn.Conv2d:
            assert name == m_name
            
            n1,n2,kw,kh = module.weight.size()
            w,idx = module.weight.flatten().sort(descending=True)
            
            j = int(mratio[-1]*w.numel())
            flat_out = m_module.weight.flatten()
            flat_out[idx[j:]] = 0
            flat_out[idx[:j]] = 1
            flat_out.reshape(n1,n2,kw,kh)
    
    out_dir = os.path.dirname(weight_pth)
    outfile = os.path.join(out_dir, save_m.format(mratio[-1]))
    torch.save(m_mask.state_dict(), outfile)

if __name__ == '__main__':
    args = m_parse()

    model = None
    if args.method == 'protonet':
        train_few_shot_params   = dict(n_way = args.train_n_way, n_support = args.n_shot) 
        model = ProtoNet(model_dict[args.model], **train_few_shot_params)
        m_mask= ProtoNet(model_dict[args.model], **train_few_shot_params)
    else:
        raise ValueError('Only support protonet!')

    model.load_state_dict(torch.load(args.weights)['state'])

    rank_mask(model, m_mask, args.mratio, args.weights, args.save_m)

