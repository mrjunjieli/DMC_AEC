#author :JunjieLi
#createtime:2021/01

from model.DMC_AEC import MT_AEC_NS_Model
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import os 
import time 
import torch.nn as nn
import logging 
import argparse
from Solver import Solver

#set the seed for generating random numbers. 
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)




def main(args,use_gpu):
    logFileName = './log/'+str(args.data_set_name)+'/train_lr'+str(args.lr)+'.log'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logFileName,mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info("***batch_size=%d***"%args.batch_size)


    model = MT_AEC_NS_Model()
    model = nn.DataParallel(model)

    if use_gpu:
        model.cuda()

    optimizer = optim.Adam([{'params':model.parameters()}],lr=args.lr,weight_decay=1e-5)

    solver = Solver(args,model=model,use_gpu=use_gpu,optimizer=optimizer,logger=logger)
    solver.train()

if __name__=='__main__':

    parser = argparse.ArgumentParser('AVConv-TasNet')
    
    #training
    parser.add_argument('--batch_size',type=int,default=4,help='Batch size')
    parser.add_argument('--num_workers',type=int,default=10,help='number of workers to generate minibatch')
    parser.add_argument('--num_epochs',type=int,default=500,help='Number of maximum epochs')
    parser.add_argument('--lr',type=float,default=1e-3,help='Init learning rate')
    parser.add_argument('--continue_from',type=str,default=None)
    parser.add_argument('--data_set_name',default=None)

    args = parser.parse_args()

    use_gpu= torch.cuda.is_available()

    os.makedirs('./log/'+args.data_set_name,exist_ok=True)


    main(args,use_gpu)
