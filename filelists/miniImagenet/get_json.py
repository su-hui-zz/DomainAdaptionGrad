import json
import os
import glob
import pdb


file = '/home/suh/deeplearning/FewShot/CloserLookFewShot_randinit/filelists/miniImagenet/images/'
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
for label, subdir in enumerate(subdirs):
    cls_name = cls_names[label]
    img_names = glob.glob(subdir+'/*.jpg')
    for img_name in img_names:
        infs['image_names'].append(img_name)
        infs['image_labels'].append(label)

f = open('./train.json','w')
json.dump(infs, f)
f.close()
