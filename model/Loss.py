import torch
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources
import numpy as np
from itertools import permutations

EPS = 1e-8


def permute_SI_SNR(_s_lists, s_lists):
    '''
        Calculate all possible SNRs according to 
        the permutation combination and 
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([cal_si_snr(s, _s)
                     for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
    return max(results)


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR without PIT training.
    Args:
        source: [B, C, T] a tensor
        estimate_source: [B, C, T] a tensor
        B:batch_size C: channel T:lenght of audio 
        in this case C==1  only single channel
    """
    # assert source.size() == estimate_source.size()
    min_length = np.min([source.shape[2], estimate_source.shape[2]])
    source = source[:, :, 0:min_length]
    estimate_source = estimate_source[:, :, 0:min_length]

    B, C, T = source.size()

    # Step 1. Zero-mean norm
    # num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / T
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate

    # Step 2. SI-SNR without PIT
    # reshape to use broadcast
    s_target = zero_mean_target        # [B, C, T]
    s_estimate = zero_mean_estimate    # [B, C, T]
    # s_target = source
    # s_estimate = estimate_source

    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # print('s_target:',pair_wise_proj[0][0][0:10])
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]

    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]

    # si_snr = torch.mean(pair_wise_si_snr)

    si_snr = torch.sum(pair_wise_si_snr, dim=0)

    return si_snr


def wSDRLoss(mixed, clean, clean_est):
    '''
    B,C,T
    '''
    def bsum(x): return torch.sum(x, dim=2)

    def mSDRLoss(orig, est):
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=2) * torch.norm(est, p=2, dim=2)
        return -(correlation / (energies + EPS))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + EPS)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * \
        mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


def cal_SDRi(src_ref, src_est):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    return sdr


if __name__ == '__main__':
    # x = torch.randn(10, 1, 48000)
    # y = torch.randn(10, 1, 48000)
    # import torchmetrics
    # metric = torchmetrics.SI_SNR()
    # metric.to('cuda')

    # x = torch.tensor([[[2.,1.]]])
    # y = torch.tensor([[[2.,1.]]])
    # acc = metric(x, y)
    # acc = metric.compute()
    # print(cal_si_snr(x,y))
    # print(acc)

    # print('snr',cal_snr(y,x).item())
    # print('snr',cal_snr_(y,x).item())
    # print('sdr',cal_sdr_(y.numpy(),x.numpy()))
    # print(x)
    x = torch.tensor([[[1., 2., 3., 4.]]])
    y = torch.tensor([[[2., 2., 2., 2.]]])
    y_ = torch.tensor([[[3., 3., 3., 3.]]])
    print(wSDRLoss(x, y_, y))
    # print(cal_si_snr(x, y))
