import sys

sys.path.append('/Work21/2020/lijunjie/AEC/model')

from conv_stft import ConvSTFT, ConviSTFT
import torch.optim as optim
from Modelutils import *
from complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm
import math
from torchsummary import summary
import cv2
import torch.nn.functional as F
import torch.nn as nn
import torch
from Normolization import *


class DCCRN_sm(nn.Module):

    def __init__(
        self,
        rnn_layers=2,
        rnn_units=128,
        win_len=400,
        win_inc=100,
        fft_len=512,
        win_type='hanning',
        masking_mode='E',
        use_clstm=False,
        use_cbn=False,
        kernel_size=5,
        kernel_num=[16, 32, 64, 128, 256, 256]
    ):
        ''' 

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(DCCRN_sm, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        #self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        #self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [4]+kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc,
                             fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc,
                               fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        # self.linear = nn.Conv2d(kernel_num[-1]*2,kernel_num[-1],3,1,1)

        self.encoder_farend = nn.ModuleList()

        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num)-1):
            self.encoder.append(
                nn.Sequential(
                    #nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(
                        self.kernel_num[idx+1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )

        hidden_dim = self.fft_len//(2**(len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim *
                        self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim *
                        self.kernel_num[-1] if idx == rnn_layers-1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim*self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(
                self.rnn_units * fac, hidden_dim*self.kernel_num[-1])

        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx]*2,
                            self.kernel_num[idx-1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(
                            self.kernel_num[idx-1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx-1]),
                        # nn.ELU()
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx]*2,
                            self.kernel_num[idx-1]//2,
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                    )
                )

        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs,farend,lens=None):
        specs = self.stft(inputs) # [batch,514,length]
        # print('specs',specs.shape)
        farend_spec = self.stft(farend)

        real = specs[:, :self.fft_len//2+1] #
        imag = specs[:, self.fft_len//2+1:]
        farend_real = farend_spec[:,:self.fft_len//2+1]
        farend_imag = farend_spec[:,self.fft_len//2+1:]

        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, farend_real, imag,farend_imag], 1) #batch,4,257,length
        # print('cspecs',cspecs.shape)
        cspecs = cspecs[:, :, 1:]   #batch,2,256,length
        # print(cspecs.shape)

        # mic_cspec = torch.stack([farend_real,farend_imag],1)

        

        # mic_cspec = mic_cspec[:,:,1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out =  cspecs 
        # out_far = mic_cspec
        
        encoder_out = []
        # encoder_out_farend = []


        for idx, layer in enumerate(self.encoder):
            # print(idx,out.shape)

            out = self.encoder[idx](out)
            # out_far = self.encoder_farend[idx](out_far)
            # print(idx,' ',out.shape)
            encoder_out.append(out)
            # encoder_out_farend.append(out_far)

        
        batch_size, channels, dims, lengths = out.size()

        # out_ = torch.cat([out,out_far],1)
        # out = self.linear(out_)

        out = out.permute(3, 0, 1, 2)
        # out_far = out_far.permute(3,0,1,2)


        if self.use_clstm:
            r_rnn_in = out[:, :, :channels//2]
            i_rnn_in = out[:, :, channels//2:]
            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels//2*dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels//2*dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])
            self.flatten_parameters()

            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels//2, dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels//2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
            # print('lstm',out.shape)

        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels*dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)
        # print('out',out.shape)

        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        #    print('decoder', out.size())
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

        if self.masking_mode == 'E':
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2(
                imag_phase,
                real_phase
            )

            #mask_mags = torch.clamp_(mask_mags,0,100)
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real, imag = real*mask_real-imag*mask_imag, real*mask_imag+imag*mask_real
        elif self.masking_mode == 'R':
            real, imag = real*mask_real, imag*mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.squeeze(out_wav, 1)
        #out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp_(out_wav, -1, 1)
        return out_spec,  out_wav



class MT_AEC_NS_Model(nn.Module):

    def __init__(self,):
        super(MT_AEC_NS_Model, self).__init__()
        
        self.echopre = DCCRN_sm(rnn_units=128,masking_mode='E',use_clstm=True,kernel_num=[128,128,128,128])

        


    def forward(self,farend,mic):

        # suppress noise
        mic_ = torch.squeeze(mic,dim=1)
        farend_= torch.squeeze(farend,dim=1)
        ptarget = self.echopre(mic_,farend_)[1]
        ptarget = torch.unsqueeze(ptarget,dim=1)



        return ptarget



def count_parameters(named_parameters):
    # Count total parameters
    total_params = 0
    part_params = {}
    for name, p in sorted(list(named_parameters)):
        n_params = p.numel()
        total_params += n_params
        part_name = name.split('.')[0]
        if part_name in part_params:
            part_params[part_name] += n_params
        else:
            part_params[part_name] = n_params

    for name, n_params in part_params.items():
        print('%s #params: %.2f M' % (name, n_params/(1024 ** 2)))
    print("Total %.2f M parameters" % (total_params / (1024 ** 2)))
    print('Estimated Total Size (MB): %0.2f' %
          (total_params * 4. / (1024 ** 2)))


if __name__ == '__main__':
    import sys
    sys.path.append('/Work21/2020/lijunjie/AEC')

    print('start')

    model = MT_AEC_NS_Model()
    # model = DCCRN(rnn_units=64,masking_mode='E',use_clstm=True,kernel_num=[16,32,64])
    model = model.cuda()

    count_parameters(model.named_parameters())
    from DataLoader import data_loader

    # data = data_loader(echo_path='../dataprocess/synthetic/data/test/echo.lst',
    #                    farend_path='../dataprocess/synthetic/data/test/far_end.lst',
    #                    nearend_path='../dataprocess/synthetic/data/test/near_end.lst',
    #                    target_path='../dataprocess/synthetic/data/test/target.lst', batch_size=1, stage='test')

    # for x in data:
    #     # print(x['idx'])
    #     farend_data = x['farend'].cuda()
    #     mic_data = x['mic'].cuda()

    #     farend_data = farend_data.to(torch.float32)
    #     mic_data = mic_data.to(torch.float32)

    #     mic_ = torch.squeeze(mic_data,dim=1)
    #     farend_data_ = torch.squeeze(farend_data,dim=1)
    #     output = model(mic_,farend_data_)
    #     # model(farend_data, mic_data)
        # break
