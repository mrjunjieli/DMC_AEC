import torch.nn as nn 
import torch 
import numpy as np 

class GlobalLayerNorm(nn.Module):
    '''

    normalize over both the channel and the time dimensions 

    gLN(F) = (F-E[F])/(Var[F]+eps)**0.5 element-wise y+beta 
    E[F] = 1/(NT)*sum_NT(F)[add elements in F along N and T dimensions] 
    Var[F] = 1/(NT)*sum_NT((F-E[F])**2)  

    N:channle dimension 
    T:time dimension 
    y and beta are trainable parameters ->R^{N*1}

    where F ->R^{N*T} 
    dim:(int or list or torch.Size) - input shape from an expected input of size
    elementwise_affine: a boolean value that when set to True, then this module has 
    learneable parameter initialized to ones(for weights) and zeros (for bias)
    '''
    def __init__(self,dim,eps=1e-05,elementwise_affine=True):
        super(GlobalLayerNorm,self).__init__()
        self.dim = dim 
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim,1))
            self.bias = nn.Parameter(torch.zeros(self.dim,1))
        else:
            self.register_parameter('weight',None)
            self.register_parameter('bias',None)

    def forward(self,x):
        if x.dim()!= 3:
            raise RuntimeError('{} accept 3D tensor as input'.format(
        self.__name__))

        mean = torch.mean(x,(1,2),keepdim=True)
        var = torch.mean((x-mean)**2,(1,2),keepdim=True)
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x 


class CumulativeLayerNorm(nn.Module):
    '''
    calculate cumulative layer normalization
    reference :https://github.com/naplab/Conv-TasNet/blob/master/utility/models.py
    '''
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(CumulativeLayerNorm, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


if __name__=='__main__':
    x = torch.rand(2,3,3)
    m = cLN(3)
    print(m(x))

