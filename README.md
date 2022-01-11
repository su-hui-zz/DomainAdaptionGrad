### Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/) before 0.4 (for newer vesion, please see issue #3 )
 - json

### Dataset
共包含CUB/CIFAR10/CIFAR100/Omniglot/miniImagenet/OfficeCaltech/DomainNet 共7种类型数据，base.json和val.json已处理好（需要更改里面的图片路径）

#### CUB
* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`
#### mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` 
(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 
#### Omniglot
* Change directory to `./filelists/omniglot`
* run `source ./download_omniglot.sh` 
#### Self-defined setting
* Require three data split json file: 'base.json', 'val.json', 'novel.json' for each dataset  
* The format should follow   
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  
See test.json for reference
* Put these file in the same folder and change data_dir['DATASETNAME'] in configs.py to the folder path  



### Train 

（各个训练脚本的开头都有训练示例，之前训练时候做的记录）

#### 随机初始化训练，获取模型梯度
```python ./train_randinit_getgrads_resnet18.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```
表示利用torchvision中自带的resnet18网络结构训练，获取模型梯度。训练过程中，每个step随机初始化模型。

For example, run `python train_randinit_getgrads_resnet18.py --method protonet --n_shot 1  --model_name randinit_resnet18 --train_aug`  
Commands below follow this example, and please refer to io_utils.py for additional options.

#### 加载预训练模型训练，获取模型梯度
```python ./train_preinit_getgrads.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] --tweights [WEIGHTPATH] --model_name [MODELSAVEPATH] [--OPTIONARG]```
表示加载预训练模型训练，获取模型梯度。
For example, run `python train_preinit_getgrads.py --model visionresnet18 --method protonet --n_shot 1 --dataset oc_clipart --tweights resnet18_vision-epoch130-acc63.483.pth --model_name clipart_resnet18_gradclipart --train_aug`  
Commands below follow this example, and please refer to io_utils.py for additional options.

### Test 

——测试域1的训练模型在域2数据上的准确率情况：（在域1数据上训练得到的模型，在域2数据上的准确率情况）

——测试操作首先保存域1的训练模型在域2数据上的特征，然后用特征计算准确率（2步操作）

#####  步骤1： Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
```python ./save_features_office_resnet18.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] --savename [FEATURESAVEPATH] --tweights [WEIGHTPATH]```
其中dataset表示域2数据；tweights表示域1模型。示例：python ./save_features.py --dataset oc_amazon --model visionresnet18 --method protonet --train_aug  --n_shot 1 --savename scratch_amazon --tweights /home/suh/deeplearning/pytrain_config/pycifar/classification_training/office_caltech_resnet18_scratch/amazon/checkpoints/resnet18_vision-epoch120-acc100.000.pth（savename为scratch_amazon，则保存在"configs.py中的save_dir/features/scratch_amazon/..."中）

##### 步骤2：test accuracy
```2. python ./test.py --dataset oc_amazon --savename scratch_amazon --model visionresnet18 --method protonet --train_aug```
表示用scratch_amazon中的模型测试数据oc_amazon的准确率情况

### 余弦相似度获取
Run
```python ./get_cos_distance.py --model [BACKBONENAME] --weights1 [WEIGHTNAME1] --weights2 [WEIGHTNAME2]```
示例：python get_cos_distance.py --model visionresnet18 --weights1 ./Save/checkpoints/oc_painting/clipart_resnet18_gradpainting_protonet_aug_5way_1shot/400.tar --weights2 ./Save/checkpoints/oc_clipart/clipart_resnet18_gradclipart_protonet_aug_5way_1shot/400.tar



### Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

### References
Our testbed builds upon several existing publicly available code. Specifically, we have modified from the following code into this project:
* title={A Closer Look at Few-shot Classification},author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
https://github.com/wyharveychen/CloserLookFewShot

### 额外说明：
   *io_utils.py中的model_dict涵盖了所用到的所有模型,里面的visionresnet18/visionvgg/visiongoogle网络结构完全来自于torvision，去除全连接层，swin和ceit来源于对应文章的官方
   *configs.py中涵盖了所用到的所有数据 
   *Train和余弦相似度获取，所有网络结构都跑过；Test部分只跑过resnet18的结构，只在Domainnet和Office两个域适应数据上操作过
