import json
import os
import glob
import pdb
import random

file = '/home/suh/dataset/OfficeCaltech/OfficeCaltechDomainAdaptation/images/amazon/'
#file = '/home/suh/dataset/OfficeCaltech/OfficeCaltechDomainAdaptation/images/caltech10/'
#file = '/home/suh/dataset/OfficeCaltech/OfficeCaltechDomainAdaptation/images/dslr/'
#file = '/home/suh/dataset/OfficeCaltech/OfficeCaltechDomainAdaptation/images/webcam/'
subdirs = glob.glob(file+'*')
infs = {}

# get cls names
cls_names = []
for subdir in subdirs:
    cls_names.append(subdir.split('/')[-1])
print("cls_num:",len(cls_names))
infs['label_names'] = cls_names

# get img paths
infs['image_names'] = []
infs['image_labels'] = []
infc = {}
for label, subdir in enumerate(subdirs):
    cls_name = cls_names[label]
    img_names = glob.glob(subdir+'/*.jpg')
    for img_name in img_names:
        infs['image_names'].append(img_name)
        infs['image_labels'].append(label)
        
        if label not in infc:
            infc[label] = [img_name]
        else:
            infc[label].append(img_name)

num_each_cls = []
max_num = 0
for cls_index in range(len(cls_names)):
    num_tcls = len(infc[cls_index])
    num_each_cls.append(num_tcls)
    if max_num < num_tcls:
        max_num = num_tcls

for cls_index in range(len(cls_names)):
    v = num_each_cls[cls_index]
    while(v<max_num):
        ind = random.randint(0,num_each_cls[cls_index]-1)
        infs['image_names'].append(infc[cls_index][ind])
        infs['image_labels'].append(cls_index)
        v+=1

for cls_index in range(len(cls_names)):
    print(cls_index,'-',infs['image_labels'].count(cls_index))

f = open('./amazon/base.json','w')
json.dump(infs, f)
f.close()
#pdb.set_trace()