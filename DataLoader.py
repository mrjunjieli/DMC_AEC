import librosa
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np 
import random
import torch.nn as nn
from scipy import spatial




def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split(" ",1)
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict


def load_audio(index_dict,index,timeLen=3,sr=16000):
    '''
    load audio data
    '''
    keys = list(index_dict.keys())
    key=''
    if type(index) not in [int, str]:
        raise IndexError('Unsupported index type: {}'.format(type(index)))
    elif type(index) ==int:
        num_uttrs = len(index_dict)
        if(num_uttrs<=index or index <0):
            raise KeyError('Interger index out of range,suppose to get 0 to {:d} \but get {:d}'.format(num_uttrs-1,index))
        key = keys[index]
    else:
        key = index 
    audio_data,sr = librosa.load(index_dict[key],sr=sr) 
    audio_data = audio_data.astype(np.float32)

    if len(audio_data)>timeLen*sr:
        audio_data = audio_data[0:timeLen*sr]
    else:
        temp = np.zeros((timeLen*sr))
        temp[0:len(audio_data)] = audio_data
        audio_data = temp

    audio_data = np.expand_dims(audio_data,axis=0)

    return audio_data 

def get_allspeakername(speakername_file='./data/M2Met/speaker_name_forAll.txt'):
    speakername_list = []
    with open(speakername_file, 'r') as f:
        for item in sorted(f):
            speakername_list.append(item.strip()) #去掉\n
    return speakername_list



class AudioDataset(Dataset):

    def __init__(self,echo_path=None,farend_path=None,nearend_path=None,
                target_path=None,stage='train',sr=16000):
        super(AudioDataset,self).__init__()
        self.echo = handle_scp(echo_path)
        self.farend = handle_scp(farend_path)
        self.nearend = handle_scp(nearend_path)
        self.target = handle_scp(target_path)

        self.sr=sr
        self.stage = stage
        if self.stage=='train':
            self.key = list(self.target.keys())

    def __len__(self):
        return len(self.echo)
    def __getitem__(self,index):
        
        echo_audio = load_audio(self.echo,index,timeLen=10,sr=self.sr)
        farend_audio = load_audio(self.farend,index,timeLen=10,sr=self.sr)

        if self.stage=='train':
            key = random.choice(self.key)
            nearend_audio = load_audio(self.nearend,key,timeLen=10,sr=self.sr)
            target_audio = load_audio(self.target,key,timeLen=10,sr=self.sr)
            seed = random.uniform(0,1)
            if seed>=0.0 and seed <0.1:
                echo_audio = np.zeros(echo_audio.shape)
                farend_audio = echo_audio
                # print('echo',echo_audio)
            elif seed>=0.1 and seed <0.2:
                nearend_audio = np.zeros(nearend_audio.shape)
                target_audio = nearend_audio
            else:
                pass
            mic_audio = echo_audio+nearend_audio
        else:
            nearend_audio = load_audio(self.nearend,index,timeLen=10,sr=self.sr)
            target_audio = load_audio(self.target,index,timeLen=10,sr=self.sr)
            mic_audio = echo_audio+nearend_audio
        
        idx = list(self.echo.keys())[index]

 

        return{
            'idx':idx,
            'echo':echo_audio,
            'farend':farend_audio,
            'nearend':nearend_audio,
            'mic':mic_audio,
            'target':target_audio,
            # 'delay':delay_time
            
        }



class AudioDataLoader(DataLoader):
    def __init__(self,*args,**kwargs):
        super(AudioDataLoader,self).__init__(*args,**kwargs)


def data_loader(echo_path=None,farend_path=None,nearend_path=None,
                target_path=None,stage='train',sr=16000,batch_size=1,num_workers=6,
                pin_memory=True,prefetch_factor=4,shuffle=True):
    dataset = AudioDataset(echo_path=echo_path,farend_path=farend_path,nearend_path=nearend_path,
                target_path=target_path,stage=stage,sr=sr)
    dataLoader = AudioDataLoader(dataset,batch_size=batch_size,shuffle=shuffle
                ,num_workers=num_workers,pin_memory=pin_memory,prefetch_factor=prefetch_factor)
    
    return dataLoader



if __name__=='__main__':
    
    x = data_loader(echo_path='./dataprocess/synthetic/data/test/echo.lst',
    farend_path='./dataprocess/synthetic/data/test/far_end.lst',
    nearend_path='./dataprocess/synthetic/data/test/near_end.lst',
    target_path='./dataprocess/synthetic/data/test/target.lst',num_workers=1,shuffle=False,stage='train')
    # print(len(x))
    
    for idx,i in enumerate(x):
        print(i['idx'])
        print('-----')
        if idx>=10:
            break
