from functools import total_ordering
import time
import torch
from model.Loss import cal_si_snr,wSDRLoss
from DataLoader import data_loader
import os
import torch.nn as nn
import math


class Solver(object):
    def __init__(self, args, model, use_gpu, optimizer, logger):
        
        self.train_loader = data_loader(echo_path='./dataprocess/synthetic/data/train/echo.lst',
                farend_path='./dataprocess/synthetic/data/train/far_end.lst',
                nearend_path='./dataprocess/synthetic/data/train/near_end.lst',
                target_path='./dataprocess/synthetic/data/train/target.lst',
                stage='train',batch_size=args.batch_size,num_workers=args.num_workers,
                prefetch_factor=10)
        self.pretrain_len= len(self.train_loader)

        self.dev_loader=data_loader(echo_path='./dataprocess/synthetic/data/test/echo.lst',
                farend_path='./dataprocess/synthetic/data/test/far_end.lst',
                nearend_path='./dataprocess/synthetic/data/test/near_end.lst',
                target_path='./dataprocess/synthetic/data/test/target.lst',
                stage='test',batch_size=args.batch_size,num_workers=args.num_workers)
        self.dev_len = len(self.dev_loader)

        

        self.args = args
        self.model = model
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.logger = logger

        self._rest()
        self.logger.info('learning rate:'+str(self.optimizer.state_dict()['param_groups'][0]['lr']))

    def _rest(self):
        self.halving = False
        if str(self.args.continue_from)!='None':
           
            checkpoint_name = str(self.args.continue_from)
            checkpoint = torch.load('./log/'+str(self.args.data_set_name)+'/model/'+checkpoint_name)

            # load model
            model_dict = self.model.state_dict()
            pretrained_model_dict = checkpoint['model']
            pretrained_model_dict = {
                k: v for k, v in pretrained_model_dict.items() if k in model_dict}
            model_dict.update(pretrained_model_dict)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("*** model "+checkpoint_name +
                             " has been successfully loaded! ***")
            # load other params
            self.start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']
            

        else:
            self.start_epoch = 0
            self.best_val_loss = float('inf')
            self.val_no_impv = 0
            # self.pre_val_sisnr = float("inf")
            self.logger.info("*** train from scratch ***")

    def train(self):
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.logger.info("------------")
            self.logger.info("Epoch:%d/%d" % (epoch, self.args.num_epochs))
            # train
            # --------------------------------------
            start = time.time()
            self.model.train()
            temp = self._run_one_epoch(self.train_loader, state='train',epoch=epoch)

            pecho_mic_loss = temp['pecho_mic_loss']/self.pretrain_len
            loss_nearend = temp['loss_nearend']/self.pretrain_len
            loss_target = temp['loss_target']/self.pretrain_len

            pecho_mic_si_snr = temp['pecho_mic_si_snr']/self.pretrain_len/self.args.batch_size
            pnearend_si_snr = temp['pnearend_si_snr']/self.pretrain_len/self.args.batch_size
            ptarget_si_snr = temp['ptarget_si_snr']/self.pretrain_len/self.args.batch_size

            end = time.time()
            self.logger.info("Train: pecho_mic_loss:%.04f,loss_nearend:%.04f,loss_target:%.04f,pecho_mic_si_snr:%.04f,pnearend_si_snr:%.04f,ptarget_si_snr:%.04f\Time:%d minutes" 
                            %(pecho_mic_loss,loss_nearend,loss_target,\
                        pecho_mic_si_snr,pnearend_si_snr,ptarget_si_snr,\
                            (end-start)//60))

            # validation
            # --------------------------------------
            start = time.time()
            self.model.eval()
            with torch.no_grad():
                temp = self._run_one_epoch(self.dev_loader, state='val')

                pecho_mic_loss = temp['pecho_mic_loss']/self.dev_len
                loss_nearend = temp['loss_nearend']/self.dev_len
                loss_target = temp['loss_target']/self.dev_len

                pecho_mic_si_snr = temp['pecho_mic_si_snr']/self.dev_len/self.args.batch_size
                pnearend_si_snr = temp['pnearend_si_snr']/self.dev_len/self.args.batch_size
                ptarget_si_snr = temp['ptarget_si_snr']/self.dev_len/self.args.batch_size

                end = time.time()
                self.logger.info("dev: pecho_mic_loss:%.04f,loss_nearend:%.04f,loss_target:%.04f,pecho_mic_si_snr:%.04f,pnearend_si_snr:%.04f,ptarget_si_snr:%.04f\Time:%d minutes" 
                                %(pecho_mic_loss,loss_nearend,loss_target,\
                            pecho_mic_si_snr,pnearend_si_snr,ptarget_si_snr,\
                                (end-start)//60))
                val_loss = -ptarget_si_snr
                self.logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            # check whether to adjust learning rate and early stop
            # -------------------------------------
            if val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 3:
                    self.halving = True
                if self.val_no_impv >= 6:
                    self.logger.info(
                        "No improvement for 6 epoches in val dataset, early stop")
                    break
            else:
                self.val_no_impv = 0

            # half the learning rate
            # -----------------------------------
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
                self.optimizer.load_state_dict(optim_state)
                self.logger.info("**learning rate is adjusted from [%f] to [%f]"
                                 % (optim_state['param_groups'][0]['lr']*2, optim_state['param_groups'][0]['lr']))
                self.halving = False


            # save the model
            # ----------------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {'model': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict(),
                              'epoch': epoch+1,
                              'best_val_loss': self.best_val_loss,
                              'val_no_impv': self.val_no_impv}
                os.makedirs('./log/'+str(self.args.data_set_name)+'/model/',exist_ok=True)
                torch.save(
                    checkpoint, "./log/"+str(self.args.data_set_name)+"/model/Checkpoint_%04d.pt" % epoch)
                self.logger.info(
                    "***save checkpoint as Checkpoint_%04d.pt***" % epoch)

    def _run_one_epoch(self, audio_data_loader, state='train',epoch=0):
        epoch_loss = {'pecho_mic_loss': 0,'pecho_ref_loss':0,
            'loss_nearend':0,'loss_target':0,'pecho_mic_si_snr':0,
            'pecho_ref_si_snr':0,'pnearend_si_snr':0,'ptarget_si_snr':0}
        length = len(audio_data_loader)

        for idx,audio in enumerate(audio_data_loader):
            echo_data = audio['echo'].to(torch.float32)
            farend_data = audio['farend'].to(torch.float32)
            mic_data = audio['mic'].to(torch.float32)
            nearend_data = audio['nearend'].to(torch.float32)
            target_data = audio['target'].to(torch.float32)

            if self.use_gpu:
                echo_data = echo_data.cuda()
                farend_data = farend_data.cuda()
                mic_data = mic_data.cuda()
                nearend_data = nearend_data.cuda()
                target_data = target_data.cuda()
           
            pecho_mic,p_nearend,p_target = self.model(farend_data,mic_data)
            # p_target = self.model(farend_data,mic_data)


            
            pecho_mic_si_snr = cal_si_snr(echo_data,pecho_mic)
            pnearend_si_snr = cal_si_snr(nearend_data,p_nearend)
            ptarget_si_snr = cal_si_snr(target_data,p_target)

            pecho_mic_loss =0
            loss_nearend=0
            loss_target=0
            audio_length = echo_data.shape[2]//10

            for i in range(10):
                pecho_mic_loss +=wSDRLoss(mic_data[:,:,i*audio_length:(i+1)*audio_length],echo_data[:,:,i*audio_length:(i+1)*audio_length],pecho_mic[:,:,i*audio_length:(i+1)*audio_length])
                loss_nearend +=wSDRLoss(mic_data[:,:,i*audio_length:(i+1)*audio_length],nearend_data[:,:,i*audio_length:(i+1)*audio_length],p_nearend[:,:,i*audio_length:(i+1)*audio_length])
                loss_target += wSDRLoss(mic_data[:,:,i*audio_length:(i+1)*audio_length],target_data[:,:,i*audio_length:(i+1)*audio_length],p_target[:,:,i*audio_length:(i+1)*audio_length])

            
            pecho_mic_loss/=10
            loss_nearend /=10
            loss_target/=10


            epoch_loss['pecho_mic_loss']  += pecho_mic_loss.item()
            epoch_loss['loss_nearend'] += loss_nearend.item()
            epoch_loss['loss_target'] += loss_target.item()
            
            epoch_loss['pecho_mic_si_snr']  += pecho_mic_si_snr.item()
            epoch_loss['pnearend_si_snr'] += pnearend_si_snr.item()
            epoch_loss['ptarget_si_snr'] += ptarget_si_snr.item()

            
            # total_loss =0.2*pecho_mic_loss+ loss_nearend+ loss_target
            total_loss = loss_target
            # total_loss = pecho_mic_loss+loss_target
            # total_loss =loss_target
            # total_loss = 0.1*pecho_mic_loss+0.2*loss_nearend+loss_target
            # total_loss = loss_nearend+ loss_target
            # total_loss = 0.2*pecho_mic_loss + 0.5*loss_nearend+ loss_target
            # total_loss = pecho_mic_loss + loss_nearend + loss_target
            



            if state == 'train':
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=20,norm_type=2)
                if math.isnan(grad_norm):
                    self.logger.info(str(idx)+"Grad norm is NAN. DO NOT UPDATE MODEL!" + str(total_loss))
                else:
                    if idx %2==0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()


                if idx %100==0:
                    # self.logger.info('alpah:'+str(alpah.item())+'beta:'+str(beta.item()))
                    self.logger.info('processed batch:%d/%d'%(idx,length))

        return epoch_loss
