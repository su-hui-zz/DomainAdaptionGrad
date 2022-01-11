'''
计算两个相同结构的网络，训练得到的grad的余弦距离
'''
import torch
import argparse
import matplotlib.pyplot as plt
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file 
import os
import pdb

#######################################################################

##################################################################
#python get_cos_distance.py --model visionresnet18 --weights1 ../CloserLookFewShot_randinit/Save/checkpoints/oc_painting/clipart_resnet18_gradpainting_protonet_aug_5way_1shot/400.tar --weights2 ../CloserLookFewShot_randinit/Save/checkpoints/oc_clipart/clipart_resnet18_gradclipart_protonet_aug_5way_1shot/400.tar

def m_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model'  , default='visionresnet18',  help='model: visionceit/visionswin/visionvgg/visionresnet18/visiongoogle') # 50 and 101 are not used in the paper
    parser.add_argument('--method' , default='protonet',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency

    parser.add_argument('--weights1', type=str, default = './Save/checkpoints/oc_amazon/amazon_sz84_resnet18_gradamazon_protonet_aug_5way_1shot/500.tar') # resnet18 CUB for protonet 
    parser.add_argument('--weights2', type=str, default = './Save/checkpoints/oc_caltech10/amazon_sz84_resnet18_gradcaltech10_protonet_aug_5way_1shot/500.tar') # resnet18 CUB for protonet 

    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--n_shot'      , default=1, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    args = parser.parse_args()
    return args


def get_cosine(model1, model2):
    smi_count = 0
    smi_sum   = 0
    for (name1, module1), (name2, module2) in zip(model1.named_modules(),model2.named_modules()):
        if type(module1) == torch.nn.Conv2d and type(module2) == torch.nn.Conv2d:
            assert name1 == name2
            n1,n2,kw,kh = module1.weight.size()
            w1 = module1.weight.flatten()
            w2 = module2.weight.flatten()
            similarity = torch.cosine_similarity(w1,w2,dim=0)
            print(name1,'-',similarity)
            smi_count += 1
            smi_sum   += similarity.item()
        elif type(module1) == torch.nn.modules.linear.Linear and type(module2) == torch.nn.modules.linear.Linear:
            # for transformer
            n1,n2 = module1.weight.size()
            w1 = module1.weight.flatten()
            w2 = module2.weight.flatten()
            similarity = torch.cosine_similarity(w1,w2,dim=0)
            print(name1,'-',similarity)
            smi_count += 1
            smi_sum   += similarity.item()
    print("avg:", smi_sum / smi_count)


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

    get_cosine(model1, model2)



