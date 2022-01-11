import json
import os
import glob
import pdb
import random

# ###################################
# amazon
# [92, 100, 99, 94, 100, 98, 82, 99, 94, 100]

# ###################################
# caltech10
# [151, 85, 138, 100, 94, 97, 110, 133, 87, 128]

# ###################################
# dslr
# [12, 10, 13, 12, 12, 23, 21, 22, 8, 24]

# ###################################
# webcam
# [29, 27, 27, 31, 30, 30, 21, 43, 27, 30]


dir = '/home/suh/dataset/OfficeCaltech/OfficeCaltechDomainAdaptation/images/'
domains = ['amazon','caltech10','dslr','webcam']
cls_names = None

infs = {}
for domain in domains:
    infs[domain] = []
    sdir = os.path.join(dir, domain)
    if cls_names == None:
        cls_names = os.listdir(sdir)

    for cls_name in cls_names:
        cdir = os.path.join(sdir, cls_name)
        if not os.path.exists(cdir):
            os.makedirs(cdir)
            infs[domain].append(0)
        else:
            cls_num = len(os.listdir(cdir))
            infs[domain].append(cls_num)

for domain in domains:
    print('\n')
    print('###################################')
    print(domain)
    print(infs[domain])


