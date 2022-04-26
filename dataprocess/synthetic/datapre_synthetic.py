import os
import librosa
import random

dataset_dir = '/CDShare3/AEC/synthetic/data'
output_dir = './data'
os.makedirs(output_dir+'/train', exist_ok=True)
os.makedirs(output_dir+'/test', exist_ok=True)

rate = 0.9  # train:dev =9:1


data_dict_echo = {}
data_dict_farend = {}
data_dict_nearend = {}
data_dict_mic = {}
data_dict_target = {}

for data in os.listdir(dataset_dir):
    data_path = os.path.join(dataset_dir, data)
    num_name = data.split('_')[0]
    category_name = data.split('_')[1].split('.')[0]

    if category_name == 'echo':
        data_dict_echo[num_name] = data_path
    elif category_name == 'farend':
        data_dict_farend[num_name] = data_path
    elif category_name == 'mic':
        data_dict_mic[num_name] = data_path
    elif category_name == 'target':
        data_dict_target[num_name] = data_path
    elif category_name == 'nearend':
        data_dict_nearend[num_name] = data_path
    else:
        raise NameError('')


# 分割训练和测试
data_key = ['f%05d' % x for x in range(10000)]
train_key = random.sample(data_key, 9000)
test_key = list(set(data_key)-set(train_key))

with open(os.path.join(output_dir,'train','echo.lst'),'w') as p:
    for key in sorted(train_key):
        p.write(key+' '+data_dict_echo[key]+'\n')
with open(os.path.join(output_dir,'train','far_end.lst'),'w') as p:
    for key in sorted(train_key):
        p.write(key+' '+data_dict_farend[key]+'\n')
with open(os.path.join(output_dir,'train','near_end.lst'),'w') as p:
    for key in sorted(train_key):
        p.write(key+' '+data_dict_nearend[key]+'\n')
with open(os.path.join(output_dir,'train','mic.lst'),'w') as p:
    for key in sorted(train_key):
        p.write(key+' '+data_dict_mic[key]+'\n')
with open(os.path.join(output_dir,'train','target.lst'),'w') as p:
    for key in sorted(train_key):
        p.write(key+' '+data_dict_target[key]+'\n')



with open(os.path.join(output_dir,'test','echo.lst'),'w') as p:
    for key in sorted(test_key):
        p.write(key+' '+data_dict_echo[key]+'\n')
with open(os.path.join(output_dir,'test','far_end.lst'),'w') as p:
    for key in sorted(test_key):
        p.write(key+' '+data_dict_farend[key]+'\n')
with open(os.path.join(output_dir,'test','near_end.lst'),'w') as p:
    for key in sorted(test_key):
        p.write(key+' '+data_dict_nearend[key]+'\n')
with open(os.path.join(output_dir,'test','mic.lst'),'w') as p:
    for key in sorted(test_key):
        p.write(key+' '+data_dict_mic[key]+'\n')
with open(os.path.join(output_dir,'test','target.lst'),'w') as p:
    for key in sorted(test_key):
        p.write(key+' '+data_dict_target[key]+'\n')
