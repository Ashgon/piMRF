"""
Created on Thu Dec 23 2025
@author: Chaoguang Gong
This module implements the piMRF reconstruction class using unsupervised learning.
"""

import os
import numpy as np
import torch
import scipy.io as scio
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from typing import Optional
import torch.nn as nn
from tqdm import tqdm
import model.model as model
import utils.utils as utils
import utils.Nufft_multi as Nufft_multi
import configs.piMRF_config as cfg

def get_current_process_gpu_memory():
    """Get GPU memory usage for the current process."""
    if not torch.cuda.is_available():
        return None, "No GPU available"
    
    try:
        # get current process ID
        pid = os.getpid()
        
        # current device
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        
        # memory info (bytes)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        
        # convert to GB
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        max_allocated_gb = max_allocated / (1024**3)
        max_reserved_gb = max_reserved / (1024**3)
        
        info = (
            f"GPU memory usage for current process (PID: {pid}):\n"
            f"  GPU device: {device_name} (device ID: {device})\n"
            f"  allocated: {allocated_gb:.4f} GB\n"
            f"  reserved: {reserved_gb:.4f} GB\n"
            f"  peak allocated: {max_allocated_gb:.4f} GB\n"
            f"  peak reserved: {max_reserved_gb:.4f} GB"
        )
        
        return info
        
    except Exception as e:
        return None, f"Failed to get memory info: {str(e)}"


class piMRF_fisp():
    def __init__(self, unsupMaxIter, unsuper_LrImg, PolyOrder, LrCsm = 0.1, Brain_mask=None, hook_nufft=Nufft_multi, **kwargs):

        # input parameters
        self.unsupMaxIter = unsupMaxIter
        self.unsuper_LrImg = unsuper_LrImg
        self.PolyOrder = PolyOrder
        self.LrCsm = LrCsm
        self.Brain_mask = Brain_mask
        self.hook_nufft = hook_nufft
        self.kwargs = kwargs

        # common parameters
        self.nRow = cfg.brain_size[0]
        self.nCol = cfg.brain_size[1]
        self.nCoil = cfg.ncoil
        self.coor_cpu = utils.build_coordinate_train(L_PE=self.nRow, L_RO=self.nCol) #[Nx,Ny,2]
        coor = torch.from_numpy(self.coor_cpu).to(dtype=cfg.dtype_float, device=cfg.device) #[Nx,Ny,2]
        self.coor = coor.reshape(-1, 2)

        # load mask_woskull
        mask_woskull = scio.loadmat(cfg.mask_for_NMSE_path)[cfg.mask_for_NMSE_name]
        mask_woskull = torch.as_tensor(mask_woskull.astype(np.float32), dtype=cfg.dtype_float, device=cfg.device)
        mask_woskull = utils.resize_image(mask_woskull)
        self.mask_woskull = mask_woskull # [Nx, Ny]

        # nabla-BlochNet model
        self.nabla_BlochNet: Optional[nn.Module] = None

        # define poly-CSM model
        self.Acoeff = None
        self.coor_mat = None
        self.optimizer_CSM = None
        self.scheduler_CSM = None

        # define DINER-based backbone model for T1, T2, PD
        self.piMRF_T1: Optional[nn.Module] = None
        self.piMRF_T2: Optional[nn.Module] = None
        self.piMRF_PD: Optional[nn.Module] = None
        self.optimizer_piMRF_T1 = None
        self.optimizer_piMRF_T2 = None
        self.optimizer_piMRF_PD = None
        self.scheduler_piMRF_T1 = None
        self.scheduler_piMRF_T2 = None
        self.scheduler_piMRF_PD = None

        # define loss function
        self.MSE_loss_function = torch.nn.MSELoss(reduction='sum')
        self.TV_loss_function = model.MYTVLoss(mode=cfg.tv_mode)

        # During initialization, check GPU memory usage
        init_GPU_info = get_current_process_gpu_memory()
        print(f'After init, GPU memory usage:\n{init_GPU_info}\n')
        with open(cfg.log_file_path, 'a') as log_file:
            log_file.write(f"After init, GPU memory usage:\n{init_GPU_info}\n")

    
    def load_pretrained_nabla_blochnet(self):
        """
        Load pretrained nabla-BlochNet model.
        """
        # create nabla-BlochNet model
        self.nabla_BlochNet = model.NablaBlochNet(hidden_size=256).to(device=cfg.device)
        
        # load pretrained weights
        self.nabla_BlochNet.load_weights(
            weight_path=cfg.nabla_BlochNet_path,
            device=cfg.device
        )
        
        # freeze parameters by default
        self.nabla_BlochNet.freeze()
                
        print("Successfully loaded pretrained nabla-BlochNet model!")
        return


    def define_poly_csm_model(self):
        self.Acoeff = torch.randn((self.PolyOrder ** 2, self.nCoil * 2), generator=torch.manual_seed(3115)).to(cfg.device).float()
        self.Acoeff.requires_grad = True
        xcoor = self.coor_cpu[:, :, 0]
        ycoor = self.coor_cpu[:, :, 1]
        coor_mat = np.zeros((self.nRow, self.nCol, self.PolyOrder ** 2))

        for i in range(0, self.PolyOrder):
            for j in range(0, self.PolyOrder):
                coor_mat[:, :, i * self.PolyOrder + j] = xcoor ** i * ycoor ** j
        self.coor_mat = torch.from_numpy(coor_mat).to(cfg.device).float()

        self.optimizer_CSM = torch.optim.Adam([self.Acoeff], lr=self.LrCsm)
        self.scheduler_CSM = lr_scheduler.StepLR(self.optimizer_CSM, step_size=cfg.lr_scheduler_CSMaps_step_size, gamma=cfg.lr_scheduler_CSMaps_gamma)
        print("Successfully define poly-CSM model!")
        return


    def define_backbone_model_withSkipCon(self):
        self.piMRF_T1 = model.DinerSiren_skipCon(hash_table_length = self.nRow * self.nCol,
                                        in_features = cfg.backbone_in_features,
                                        hidden_features = cfg.backbone_hidden_features,
                                        hidden_layers = cfg.backbone_hidden_layers,
                                        out_features = cfg.backbone_out_features,
                                        outermost_linear=True,).to(device=cfg.device)
        self.piMRF_T2 = model.DinerSiren_skipCon(hash_table_length = self.nRow * self.nCol,
                                        in_features = cfg.backbone_in_features,
                                        hidden_features = cfg.backbone_hidden_features,
                                        hidden_layers = cfg.backbone_hidden_layers,
                                        out_features = cfg.backbone_out_features,
                                        outermost_linear=True,).to(device=cfg.device)
        self.piMRF_PD = model.DinerSiren_skipCon(hash_table_length = self.nRow * self.nCol,
                                        in_features = cfg.backbone_in_features,
                                        hidden_features = cfg.backbone_hidden_features,
                                        hidden_layers = cfg.backbone_hidden_layers,
                                        out_features = cfg.backbone_out_features,
                                        outermost_linear=True,).to(device=cfg.device)

        self.optimizer_piMRF_T1 = torch.optim.Adam(params=self.piMRF_T1.parameters(), lr=self.unsuper_LrImg)
        self.scheduler_piMRF_T1 = lr_scheduler.StepLR(self.optimizer_piMRF_T1, step_size=cfg.lr_scheduler_ParaMaps_step_size, gamma=cfg.lr_scheduler_ParaMaps_gamma)

        self.optimizer_piMRF_T2 = torch.optim.Adam(params=self.piMRF_T2.parameters(), lr=self.unsuper_LrImg)
        self.scheduler_piMRF_T2 = lr_scheduler.StepLR(self.optimizer_piMRF_T2, step_size=cfg.lr_scheduler_ParaMaps_step_size, gamma=cfg.lr_scheduler_ParaMaps_gamma)

        self.optimizer_piMRF_PD = torch.optim.Adam(params=self.piMRF_PD.parameters(), lr=self.unsuper_LrImg)
        self.scheduler_piMRF_PD = lr_scheduler.StepLR(self.optimizer_piMRF_PD, step_size=cfg.lr_scheduler_ParaMaps_step_size, gamma=cfg.lr_scheduler_ParaMaps_gamma)

        backboneInfo = "Successfully define backbone model with skip connection!\n"
        print(backboneInfo)
        with open(cfg.log_file_path, 'a') as log_file:
                    log_file.write(backboneInfo)
        return


    def main(self, ksp_und):

        writer = SummaryWriter(cfg.result_archive_root + '/logs/{}'.format(str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))
        
        self.hook_nufft.csm = self.hook_nufft.csm * self.Brain_mask[None, None, :, :] # [1, ncoils, Nx, Ny]
        ksp_und_tensor = torch.from_numpy(ksp_und).to(dtype=cfg.dtype_complex, device=cfg.device) # [L, ncoils, num_samples]

        # Get 1st egienvector subspace of k-space data
        tstDsKsp_full_tensor, Vk = utils.kSVD_get_compressed_ksp_tensor(ksp_und_tensor, Nufft_multi_handle=self.hook_nufft) # compress k-space via SVD, get full k-space tensor
        invivo_reconSigs = self.hook_nufft.adjoint_multi_op(tstDsKsp_full_tensor, smap=None, ktraj=self.hook_nufft.ktraj_full[:cfg.rank, ...], dcomp=self.hook_nufft.dcomp_full[:cfg.rank, ...]) # np.complex64 [ncoil, Nx, Ny, rank] reconstructed eigenvector subspace coil images
        invivo_reconSigs_1st_mulco = invivo_reconSigs[:, :, :, 0].squeeze() # [ncoil, Nx, Ny]

        # load pretrained nabla-BlochNet
        self.load_pretrained_nabla_blochnet()

        invivo_ATb_mulcoils = self.hook_nufft.adjoint_multi_op(ksp_und_tensor, smap=None, ktraj=self.hook_nufft.ktraj_und, dcomp=self.hook_nufft.dcomp_und) # np.complex64 [ncoil, Nx, Ny, L] reconstructed aliased coil images
        invivo_ATb_mulcoils = invivo_ATb_mulcoils.permute(3, 0, 1, 2) # [L, ncoil, Nx, Ny]
        invivo_ATb_mulcoils = invivo_ATb_mulcoils * self.Brain_mask[None, None, :, :] # apply brain mask to reconstructed images
        
        del ksp_und_tensor

        # define poly-CSM model
        self.define_poly_csm_model() 

        # define DINER-based backbone model with skip connection
        self.define_backbone_model_withSkipCon()

        self.piMRF_T1.train()
        self.piMRF_T2.train()
        self.piMRF_PD.train()

        # Check GPU memory usage before training
        before_train_GPU_info = get_current_process_gpu_memory()
        print(f'Done model define, before training, GPU memory usage:\n{before_train_GPU_info}\n')
        with open(cfg.log_file_path, 'a') as log_file:
            log_file.write(f"\nDone model define, before training, GPU memory usage:\n{before_train_GPU_info}\n")    

        print('Unsupervised piMRF Reconstruction start...')
        unsuper_iter_loop = tqdm(range(self.unsupMaxIter))        
        losses = []

        for ite in unsuper_iter_loop:
            # estimate csm using polynomial
            Acoeff_temp = self.Acoeff.unsqueeze(0).unsqueeze(0)
            pre_CSM_real = torch.sum(self.coor_mat.unsqueeze(-1) * Acoeff_temp[:, :, :, :self.nCoil], 2)
            pre_CSM_imag = torch.sum(self.coor_mat.unsqueeze(-1) * Acoeff_temp[:, :, :, self.nCoil:], 2)
            pre_CSM = torch.complex(pre_CSM_real, pre_CSM_imag)
            pre_CSM_norm = torch.sqrt(torch.sum(pre_CSM.conj() * pre_CSM, dim=2)).unsqueeze(-1)
            pre_CSM = pre_CSM / pre_CSM_norm
            pre_CSM = pre_CSM.unsqueeze(0).permute(0, 3, 1, 2) #[1,ncoils,Nx,Ny]

            # predict T1, T2, PD parameter maps
            pre_Params_T1 = self.piMRF_T1(self.coor).reshape(self.nRow, self.nCol)
            pre_Params_T2 = self.piMRF_T2(self.coor).reshape(self.nRow, self.nCol)
            pre_Params_PD = self.piMRF_PD(self.coor).reshape(self.nRow, self.nCol)

            mainsteam_pre_PD = ((pre_Params_PD + 1) / 2 * (cfg.PD_max - cfg.PD_min) + cfg.PD_min ) * self.Brain_mask # [Nx, Ny]
            non_zero_mask = (self.Brain_mask != 0) & (mainsteam_pre_PD != 0) # [Nx, Ny]

            pre_nonzero_T1T2_stack = torch.stack([pre_Params_T1[non_zero_mask], pre_Params_T2[non_zero_mask]], dim=1) # [non_zero_nvoxels, 2]            
            net_output = self.nabla_BlochNet(pre_nonzero_T1T2_stack)  # [non_zero_nvoxels, L]
            pre_MultiImg_woPD = torch.zeros((self.nRow, self.nCol, cfg.para_length), dtype=cfg.dtype_complex, device=cfg.device)
            pre_MultiImg_woPD[non_zero_mask] = net_output.to(cfg.dtype_complex)
            pre_MultiImg = pre_MultiImg_woPD * mainsteam_pre_PD.unsqueeze(-1) # [Nx, Ny, L]

            # estimate csm in temporal subspace
            pre_MultiImg_reviewed = pre_MultiImg.view(-1, cfg.para_length) # [Nx*Ny, L]
            pre_MultiImg_compressed = torch.matmul(pre_MultiImg_reviewed, Vk) # [Nx*Ny, rank]
            pre_MultiImg_Sigs = pre_MultiImg_compressed.view(self.nRow, self.nCol, -1) # [Nx, Ny, rank]
            pre_MultiImg_Sigs_1st = pre_MultiImg_Sigs[:, :, 0].squeeze() # [Nx, Ny]
            pre_MultiImg_Sigs_1st_mask = (pre_MultiImg_Sigs_1st != 0) & (self.Brain_mask != 0) # [Nx, Ny]

            sig_ratio_inv_pre = invivo_reconSigs_1st_mulco / (pre_MultiImg_Sigs_1st[None, :, :] + 1e-8) # [ncoils, Nx, Ny] csm*PD_max*ρ
            sig_ratio_inv_pre_l2norm = torch.norm(sig_ratio_inv_pre, p=2, dim=0) # [Nx, Ny] (csm*PD_max*ρ)
            cacul_csm = sig_ratio_inv_pre / (sig_ratio_inv_pre_l2norm.unsqueeze(0) + 1e-8) # [ncoils, Nx, Ny] csm
            cacul_csm = cacul_csm.unsqueeze(0)
            cacul_csm_masked = cacul_csm * pre_MultiImg_Sigs_1st_mask[None, None, :, :] # [1, ncoils, Nx, Ny]
            tstCsm_tensor_masked = pre_CSM * pre_MultiImg_Sigs_1st_mask[None, None, :, :] # [1, ncoils, Nx, Ny]
            
            # coil combination of ATb
            inv_ata_x_sysone = torch.sum(invivo_ATb_mulcoils * tstCsm_tensor_masked.conj(), dim=1, keepdim=False) # [L, Nx, Ny]
            inv_ata_x_ori = inv_ata_x_sysone.permute(1, 2, 0) # [Nx,Ny,L]
            inv_ata_x = inv_ata_x_ori # * pre_MultiImg_Sigs_1st_mask[:, :, None] # masked aliased image

            # predict image data
            pre_toep_x_ori = self.hook_nufft.ata_multi_op(pre_MultiImg, smap=tstCsm_tensor_masked, Is_multi=False) # [Nx,Ny,L]
            pre_toep_x = pre_toep_x_ori # * pre_MultiImg_Sigs_1st_mask[:, :, None] # masked reconstructed image


            dc_loss = self.MSE_loss_function(
                torch.view_as_real(pre_toep_x).float(), 
                torch.view_as_real(inv_ata_x).float()) * cfg.dc_weight
            
            dc_loss_csm = self.MSE_loss_function(
                torch.view_as_real(cacul_csm_masked).float(), 
                torch.view_as_real(tstCsm_tensor_masked).float()) * cfg.csm_weight

            # TV loss
            tv_loss_t1 = cfg.tv_weight[0] * self.TV_loss_function(pre_Params_T1 * self.Brain_mask, mask=self.Brain_mask)           
            tv_loss_t2 = cfg.tv_weight[1] * self.TV_loss_function(pre_Params_T2 * self.Brain_mask, mask=self.Brain_mask)            
            tv_loss_pd = cfg.tv_weight[2] * self.TV_loss_function(pre_Params_PD * self.Brain_mask, mask=self.Brain_mask)

            tv_loss = tv_loss_t1 + tv_loss_t2 + tv_loss_pd      
            loss = dc_loss + tv_loss + dc_loss_csm
            
            self.optimizer_piMRF_T1.zero_grad()
            self.optimizer_piMRF_T2.zero_grad()
            self.optimizer_piMRF_PD.zero_grad()
            self.optimizer_CSM.zero_grad()

            loss.backward()

            self.optimizer_piMRF_T1.step()
            self.optimizer_piMRF_T2.step()
            self.optimizer_piMRF_PD.step()
            self.optimizer_CSM.step()

            self.scheduler_piMRF_T1.step()
            self.scheduler_piMRF_T2.step()
            self.scheduler_piMRF_PD.step()
            self.scheduler_CSM.step()
            
            losses.append([loss.item(), 
                            dc_loss.item(),
                            tv_loss.item(),
                            dc_loss_csm.item(),
                            ])

            unsuper_iter_loop.set_postfix(loss=loss.item())
            writer.add_scalar('loss', loss.item(), ite + 1)

            if ite % cfg.img_save_pre == cfg.img_save_pre-1:
                # save intermediate results and log
                with torch.no_grad():
                    pre_T1, pre_T2, pre_PD = utils.denormalize_to_oriRange(pre_Params_T1.detach(), pre_Params_T2.detach(), pre_Params_PD.detach())

                utils.show_tissue_maps(pre_T1*pre_MultiImg_Sigs_1st_mask, pre_T2*pre_MultiImg_Sigs_1st_mask, pre_PD*pre_MultiImg_Sigs_1st_mask, pre_MultiImg_Sigs_1st_mask, cfg.temp_img_save_boot, ite, losses)
                utils.show_sensitivity_maps(tstCsm_tensor_masked.detach(), cfg.temp_img_save_boot, ite)
                
                # Log GPU memory info
                GPU_info = get_current_process_gpu_memory()
                
                log_message = (f"\n\n----- Iteration {ite}: --------------------> \n"
                            f"dc_loss={dc_loss.item()}, csm_loss={dc_loss_csm.item()}, tv_loss={tv_loss.item()}\n"
                            f"GPU_memory_info:\n{GPU_info}\n")
                
                print(log_message)
                with open(cfg.log_file_path, 'a') as log_file:
                    log_file.write(log_message)
            
            if ite % cfg.backup_interval == cfg.backup_interval-1:
                # save intermediate results
                with torch.no_grad():
                    pre_T1, pre_T2, pre_PD = utils.denormalize_to_oriRange(pre_Params_T1.detach(), pre_Params_T2.detach(), pre_Params_PD.detach())

                pre_para_maps_stack = torch.stack([pre_T1, pre_T2, pre_PD], dim=2)
                filename = cfg.unique_flag + '_in_process_' + str(ite) + '.mat'
                filepath = os.path.join(cfg.result_archive_root, filename)

                # save pre_T1, pre_T2, pre_PD and pre_csm to a file
                scio.savemat(filepath, {
                    'para_maps': pre_para_maps_stack.detach().cpu().numpy(),
                    'csmaps': tstCsm_tensor_masked.detach().cpu().numpy()})

        with torch.no_grad():
            pre_T1, pre_T2, pre_PD = utils.denormalize_to_oriRange(pre_Params_T1.detach(), pre_Params_T2.detach(), pre_Params_PD.detach())
        pre_para_maps_tensor = torch.stack([pre_T1, pre_T2, pre_PD], dim=2)

        return pre_MultiImg.detach().cpu().numpy(), pre_para_maps_tensor.detach().cpu().numpy(), non_zero_mask.detach().cpu().numpy(), tstCsm_tensor_masked.detach().cpu().numpy()
