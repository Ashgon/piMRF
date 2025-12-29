#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 2025
@author: Chaoguang Gong
This script is a main program for piMRF reconstruction using unsupervised learning.
"""

import os
import time
import torch
import numpy as np
import scipy.io as sio
import model.piMRF as piMRF
import utils.utils as utils
import configs.piMRF_config as cfg
import utils.Nufft_multi as Nufft_multi

if __name__ == '__main__':

    #%% Load variables related to acquisition
    # load sequence
    mat_data = sio.loadmat(cfg.FA_TR_path)
    FAs_original = mat_data[cfg.FA_type]
    TRs_original = mat_data[cfg.TR_type]
    
    alpha = FAs_original[:cfg.para_length+1] # Flip angle (degree)    
    TRs = TRs_original[:cfg.para_length+1] # Repetition times (ms)
    TEs = np.ones(cfg.para_length+1) * cfg.TE # Echo time (ms)

    # init nufft operator
    Nufft_multi.init_nufft_multi_op()
    hook_nufft = Nufft_multi

    # generate dict
    signal_generator = utils.SIMU_EPG(fft_hook=hook_nufft, alpha=alpha, TRs=TRs, TEs=TEs)
    cfg.dic_m, cfg.LUT = signal_generator.Dic_gen()
    cfg.dic_m = torch.as_tensor(cfg.dic_m, dtype=cfg.dtype_complex, device=cfg.device)
    cfg.LUT = torch.as_tensor(cfg.LUT, dtype=cfg.dtype_float, device=cfg.device)
    
    #%% Load tissue map    
    SimuBrainMaps = sio.loadmat(cfg.data_path) # Load brainweb data
    
    # extract the T1, T2, PD maps
    BrainSlice = SimuBrainMaps[cfg.data_name]
    T1_gt = BrainSlice[:, :, 0]
    T2_gt = BrainSlice[:, :, 1]
    PD_gt = BrainSlice[:, :, 2]
    PD_gt = PD_gt / np.max(PD_gt)

    T1_gt = torch.as_tensor(T1_gt.astype(np.float32), dtype=cfg.dtype_float, device=cfg.device)
    T2_gt = torch.as_tensor(T2_gt.astype(np.float32), dtype=cfg.dtype_float, device=cfg.device)
    PD_gt = torch.as_tensor(PD_gt.astype(np.float32), dtype=cfg.dtype_float, device=cfg.device)

    T1_gt = utils.resize_image(T1_gt)
    T2_gt = utils.resize_image(T2_gt)
    PD_gt = utils.resize_image(PD_gt)
    
    brain_mask = torch.stack([(T1_gt > 0).to(cfg.dtype_float), 
                                (T2_gt > 0).to(cfg.dtype_float), 
                                (PD_gt > 0).to(cfg.dtype_float)], dim=2)
    single_mask = brain_mask[:, :, 0].squeeze()
    para_maps_gt = torch.stack([T1_gt * single_mask, T2_gt * single_mask, PD_gt * single_mask], dim=2)

    #%% Generate k-space data    
    multiImg_gt, ksp_gt_woNoise = signal_generator.KspaceUnd(T1_gt, T2_gt, PD_gt, Brain_mask=brain_mask, Pre_csm=hook_nufft.csm) # img [Nx, Ny, L]; ksp [L, ncoils, num_samples];

    # add noise
    if cfg.is_noisy:
        ksp_gt_wiNoise = utils.AWGN(ksp_gt_woNoise, cfg.undksp_snr) # add Gaussian white noise to k-space data
        test_snr = utils.calculate_snr(ksp_gt_woNoise, ksp_gt_wiNoise)
        print('The SNR of the simulated data is ',test_snr)
    else:
        ksp_gt_wiNoise = ksp_gt_woNoise
        print('The simulated data is noiseless.')
    del ksp_gt_woNoise
    
    time_all_start = time.time()
    solver = piMRF.piMRF_fisp(unsupMaxIter=cfg.unsuper_Iter_num,
                              unsuper_LrImg=cfg.unsuper_LrRate,
                              PolyOrder=8, LrCsm=0.6,
                              Brain_mask=single_mask,
                              hook_nufft=hook_nufft)
    
    pre_X, pre_para_maps, pre_mask, csmaps = solver.main(ksp_gt_wiNoise)

    time_all_end = time.time()
    duration = time_all_end - time_all_start
    minutes = duration / 60.0
    runtime_str = f"Reconstruction process took {minutes:.2f} minutes"
    print(runtime_str)
    
    # specify output path
    filename = f"{cfg.unique_flag}.mat"
    filepath = os.path.join(cfg.result_archive_root, filename)

    # save pre_T1, pre_T2, pre_PD and pre_csm into a single file
    sio.savemat(filepath, {
        'para_maps': pre_para_maps,
        'csmaps': csmaps})
    
    mask_woskull = sio.loadmat(cfg.mask_for_NMSE_path)[cfg.mask_for_NMSE_name]
    mask_woskull = torch.as_tensor(mask_woskull.astype(np.float32), dtype=cfg.dtype_float, device=cfg.device)
    mask_woskull = utils.resize_image(mask_woskull)
    mask_woskull = torch.stack([mask_woskull, mask_woskull, mask_woskull], dim=2)
    gt_brain_mask = mask_woskull
    gt_brain_mask_np = gt_brain_mask.detach().cpu().numpy()
    
    pre_T1, pre_T2, pre_PD = pre_para_maps[:, :, 0], pre_para_maps[:, :, 1], pre_para_maps[:, :, 2]
    pre_T1, pre_T2, pre_PD = [x * gt_brain_mask_np[:, :, i] for i, x in enumerate([pre_T1, pre_T2, pre_PD])]
    T1_gt, T2_gt, PD_gt = [x * gt_brain_mask[:, :, i] for i, x in enumerate([T1_gt, T2_gt, PD_gt])]
    
    nmse_T1 = utils.calculate_nmse(T1_gt, pre_T1)
    nmse_T2 = utils.calculate_nmse(T2_gt, pre_T2)
    nmse_PD = utils.calculate_nmse(PD_gt, pre_PD)    
    
    nmse_str = 'NMSE_T1: {:.4f}, NMSE_T2: {:.4f}, NMSE_PD: {:.4f}'.format(nmse_T1, nmse_T2, nmse_PD)
    psnrRecImg = utils.psnr(pre_X, multiImg_gt)
    print('PSNR: {:.4f}, {}'.format(psnrRecImg, nmse_str))

    # write NMSE values to the log file
    with open(cfg.log_file_path, 'a') as log_file:
        log_file.write('---------Runtime:' + runtime_str + '---------\n')
        log_file.write('---------NMSE:' + nmse_str + '---------\n')
        log_file.write('PSNR: {:.4f}\n'.format(psnrRecImg))
    print('NMSE values logged to:', cfg.log_file_path)

    print('piMRF reconstruction completed and results saved.')
