"""
Created on Thu Dec 23 2025
@author: Chaoguang Gong
This module implements multi-batch NUFFT operations for MRI reconstruction.
"""

import os
import torch
import numpy as np
import scipy.io as scio
import torchkbnufft as tkbn
import utils.utils as utils
import configs.piMRF_config as cfg
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_batch_index(LEN, batch_size): 
    batch_size = int(batch_size)
    num = int(LEN // batch_size)
    yu = int(LEN % batch_size)
    out = []
    if LEN == 0:
        return out
    elif LEN <= batch_size:
        out.append([0, LEN])
        return out
    for i in range(num):
        out.append([i*batch_size, (i+1)*batch_size])
    if yu != 0:
        out.append([num*batch_size, LEN])
    return out

def get_batch_index_tensor(LEN, batch_size):
    """
    Compute block index ranges.

    Parameters:
    - LEN: total length (can be NumPy array, PyTorch tensor or scalar).
    - batch_size: size of each block (can be NumPy array, PyTorch tensor or scalar).

    Returns:
    - out: list of index ranges for each block, each element is [start, end].
    """
    # Convert inputs to integers if they are arrays or tensors
    if isinstance(LEN, (np.ndarray, torch.Tensor)):
        LEN = int(LEN.item() if isinstance(LEN, torch.Tensor) else LEN.item())
    if isinstance(batch_size, (np.ndarray, torch.Tensor)):
        batch_size = int(batch_size.item() if isinstance(batch_size, torch.Tensor) else batch_size.item())

    # calculate number of full batches and remainder
    batch_size = int(batch_size)   
    num = int(LEN // batch_size)
    yu = int(LEN % batch_size)
    out = []

    if LEN == 0:
        return out
    elif LEN <= batch_size:
        out.append([0, LEN])
        return out

    for i in range(num):
        out.append([i * batch_size, (i + 1) * batch_size])

    if yu != 0:
        out.append([num * batch_size, LEN])

    return out


def init_nufft_multi_op():
    '''
    Initialize parameters for multi-batch NUFFT operators.

    Inputs (via config):
    - batch_size: batch size for time frames
    - im_size: image size (Nx, Ny)
    - L: temporal length
    - grid_factor: oversampling ratio (default 2)
    - ktraj_und: k-space trajectory (numpy array [L, 2, N_samples])

    Returns:
    - None (initializes global operator objects)
    '''
    global nufft_ob, im_size, ktraj_und, dcomp_und, csm, norm, nufft_batch_index_und, adjnufft_ob, toep_op, kernel

    """Common settings"""
    grid_factor_init = cfg.grid_factor # oversampling ratio, default 2
    numpoints = cfg.numpoints # interpolation neighbors number, default 6
    ncoil_init = cfg.ncoil # number of coils
    norm = cfg.norm # normalization method
    im_size = (cfg.brain_size[0], cfg.brain_size[1]) # image size (Nx, Ny)
    grid_size = (cfg.brain_size[0]*grid_factor_init, cfg.brain_size[1]*grid_factor_init) # grid_size
    L = cfg.para_length # temporal length
    batch_size_und = cfg.torch_batch_size_L # batch size for undersampled time frames
    batch_size_full = cfg.torch_batch_size_k # batch size for pseudo fully sampled subspace time frames
    try:
        csm = torch.from_numpy(scio.loadmat(cfg.csm_map_path)[cfg.csm_map_name]).type(cfg.dtype_complex).to(cfg.device) # [1, ncoil, Nx, Ny], torch.complex64
        csm = csm[:, :ncoil_init, :, :]
        csm = utils.resize_image(csm, target_size=cfg.brain_size)
    except FileNotFoundError:
        print('csm map not found, using default csm map.')
        csm = torch.ones((1, ncoil_init, im_size[0], im_size[1]), dtype=cfg.dtype_complex, device=cfg.device)
    
    """Nufft undersampled settings"""
    # Undersampled k-space trajectory initialization
    ktraj_und = scio.loadmat(cfg.ktraj_und_path)[cfg.ktraj_und_name]*2*np.pi # complex k-trajectory, rescale to [-pi, pi]
    ktraj_und = torch.from_numpy(ktraj_und[cfg.device_config['trajStartPoint']:cfg.device_config['trajEndPoint'],:]).type(cfg.dtype_complex).to(cfg.device)
    ktraj_und = utils.process_trajectory(ktraj_und, verbose=False) # process trajectory (periodic extension and ensure coords in [-pi, pi])
    ktraj_und = torch.stack([ktraj_und.real, ktraj_und.imag], dim=-1) # convert to real k-space trajectory [num_samples, L, 2]
    ktraj_und = ktraj_und.permute(1, 2, 0) # reshape to [2, L, num_samples]
    cfg.num_samples = int(ktraj_und.shape[2]) # number of undersampled k-space samples, general setting
    
    dcomp_und = tkbn.calc_density_compensation_function(
            ktraj=ktraj_und[0:1,:,:], im_size=im_size).to(cfg.device) # [1, 1, N_samples] compute density compensation function
    dcomp_und = dcomp_und.repeat(L, 1, 1)
    # compute batch indices for NUFFT over time frames
    if batch_size_und >= L:
        nufft_batch_index_und = [[0, L]]
    else:
        nufft_batch_index_und = get_batch_index(L, batch_size_und)

    """Subspace fully-sampled settings"""
    global ktraj_full, dcomp_full, ktraj_full_sorted_index, nufft_batch_index_full
    # Fully-sampled k-space trajectory initialization
    ktraj_selected = ktraj_und[:cfg.num_interleafs] # select num_interleafs interleaves
    ktraj_full_single = ktraj_selected.permute(0, 2, 1).reshape(-1, 2)
    ktraj_full_single_magnitude = torch.sqrt(torch.sum(ktraj_full_single ** 2, dim=1))  # [num_interleafs * num_samples] compute magnitude of k-space trajectory
    ktraj_full_sorted_index = torch.argsort(ktraj_full_single_magnitude, descending=False)  # [num_interleafs * num_samples] sort indices by magnitude
    ktraj_full_single_sorted = ktraj_full_single[ktraj_full_sorted_index, :]  # [num_interleafs * num_samples, 2] sort k-space trajectory
    ktraj_full = ktraj_full_single_sorted.unsqueeze(0).repeat(L, 1, 1)  # [L, num_interleafs * num_samples, 2] expand k-space trajectory to all time points L
    ktraj_full = ktraj_full.permute(0, 2, 1)  # [L, 2, num_interleafs * num_samples] transpose k-space trajectory
    dcomp_full = tkbn.calc_density_compensation_function(
        ktraj=ktraj_full[0:1,:,:], im_size=im_size).to(cfg.device)  # [L, 1, num_interleafs * num_samples] compute density compensation function
    dcomp_full = dcomp_full.repeat(ktraj_full.shape[0], 1, 1)  # [L, 1, num_interleafs * num_samples] replicate density compensation function
    # compute batch indices for full NUFFT over subspace rank
    if batch_size_full >= cfg.rank:
        nufft_batch_index_full = [[0, cfg.rank]]
    else:
        nufft_batch_index_full = get_batch_index(cfg.rank, batch_size_full)

    """NUFFT operator initialization"""
    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(cfg.device)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(cfg.device)
        
    # calculate Toeplitz kernel for a single cycle
    kernel_single_cycle = tkbn.calc_toeplitz_kernel(ktraj_und[:cfg.num_interleafs], im_size, weights=dcomp_und, norm=norm, grid_size=grid_size, numpoints=numpoints).to(cfg.device)
    # expand the single-cycle kernel to cover all time frames
    num_cycles = L // cfg.num_interleafs
    remainder = L % cfg.num_interleafs
    kernel = torch.cat([kernel_single_cycle] * num_cycles + [kernel_single_cycle[:remainder]], dim=0).to(cfg.device)
    toep_op = tkbn.ToepNufft().to(cfg.device) # instantiate Toeplitz NUFFT operator
    print('nufft operator initialized')

def nufft_forward_multi_op(x=None, smap=None, ktraj=None):
    '''
    NUFFT undersampled forward operation

    Inputs:
    - x: image array [Nx, Ny, L]

    Returns:
    - k_samples: k-space samples [L, ncoil, N_samples]
    '''
    L = x.shape[2]
    if L == cfg.para_length:
        nufft_batch_index = nufft_batch_index_und
    else:
        nufft_batch_index = nufft_batch_index_full

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).type(cfg.dtype_complex).to(cfg.device)
    x = x.unsqueeze(0).permute(3, 0, 1, 2) #[L,1,Nx,Ny]

    if smap is not None:
        if not isinstance(smap, torch.Tensor):
            smap = torch.tensor(smap).type(cfg.dtype_complex).to(cfg.device)#[1,ncoils,Nx,Ny]
        else:
            smap = smap.type(cfg.dtype_complex).to(cfg.device)

    if ktraj is None:
        ktraj = ktraj_und  # if no ktraj provided, use global variable ktraj_und

    k_samples = torch.zeros((L, cfg.ncoil, ktraj.shape[2]), dtype=cfg.dtype_complex, device=cfg.device) #[L,ncoils,num_samples]
    for i in nufft_batch_index:
        im_k_samples = nufft_ob(x[i[0]:i[1], :, :, :],
                                ktraj[i[0]:i[1], :, :], smaps=smap, norm=norm) #
        k_samples[i[0]:i[1], :, :] = im_k_samples

    return k_samples

def nufft_adjoint_multi_op(k_samples=None, smap=None, ktraj=None, dcomp=None):
    '''
    NUFFT adjoint (iNUFFT) reconstruction

    Inputs:
    - k_samples: k-space samples [L, ncoil, N_samples]

    Returns:
    - x: reconstructed image [Nx, Ny, L] or multi-coil [ncoil, Nx, Ny, L]
    '''    
    L = k_samples.shape[0]
    if L == cfg.para_length:
        nufft_batch_index = nufft_batch_index_und
    else:
        nufft_batch_index = nufft_batch_index_full

    if not isinstance(k_samples, torch.Tensor):
        k_samples = torch.tensor(k_samples).type(cfg.dtype_complex).to(cfg.device)

    if smap is not None:
        if not isinstance(smap, torch.Tensor):
            smap = torch.tensor(smap).type(cfg.dtype_complex).to(cfg.device) # [1,ncoils,Nx,Ny]
        else:
            smap = smap.type(cfg.dtype_complex).to(cfg.device)
        x = torch.zeros((im_size[0], im_size[1], L), dtype=cfg.dtype_complex, device=cfg.device)
    else:
        x = torch.zeros((cfg.ncoil, im_size[0], im_size[1], L), dtype=cfg.dtype_complex, device=cfg.device)

    if ktraj is None:
        ktraj = ktraj_und # If no ktraj provided, use global variable ktraj_und
    if dcomp is None:
        dcomp = dcomp_und # If no dcomp provided, use global variable dcomp_und

    for i in nufft_batch_index:
        im_x = adjnufft_ob(k_samples[i[0]:i[1], :, :]*dcomp[i[0]:i[1], :, :],
                        ktraj[i[0]:i[1], :, :], smaps=smap, norm=norm)
        x[..., i[0]:i[1]] = torch.squeeze(im_x.permute(1, 2, 3, 0))

    return x

def nufft_toep_ata_multi_op(x=None, smap=None, Is_multi=False):
    '''
    NUFFT Toeplitz ATA multi-op

    Inputs:
    - x: image array [Nx, Ny, L]

    Returns:
    - x_ata: result of A^T A applied to x, same shape as input
    '''
    L = x.shape[2]  
    if L == cfg.para_length:
        nufft_batch_index = nufft_batch_index_und
    else:
        nufft_batch_index = nufft_batch_index_full

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).type(cfg.dtype_complex).to(cfg.device)
    x = x.unsqueeze(0).permute(3, 0, 1, 2) #[L,1,Nx,Ny]
    if Is_multi:
        x_ata = torch.zeros((cfg.ncoil, im_size[0], im_size[1], L), dtype=cfg.dtype_complex, device=cfg.device) #[ncoils,Nx,Ny,L]
    else:
        x_ata = torch.zeros((im_size[0], im_size[1], L), dtype=cfg.dtype_complex, device=cfg.device) #[Nx,Ny,L]

    if smap is not None:
        if not isinstance(smap, torch.Tensor):
            smap = torch.tensor(smap).type(cfg.dtype_complex).to(cfg.device)#[1,ncoils,Nx,Ny]
        else:
            smap = smap.type(cfg.dtype_complex).to(cfg.device)
        smap = smap.expand(L, -1, -1, -1)  # [L, ncoils, Nx, Ny]        
    else:
        smap = torch.ones((L, 1, im_size[0], im_size[1])).type(cfg.dtype_complex).to(cfg.device)
        print("Notice: currently using an all-ones coil sensitivity map (smap). To use a real csm, pass smap as an argument when calling.")
        
    for i in nufft_batch_index:
            im_x = toep_op(x[i[0]:i[1], ...], kernel[i[0]:i[1], :, :], smaps=smap[i[0]:i[1], :, :, :], norm=norm, Is_multi=Is_multi) 
            x_ata[..., i[0]:i[1]] = torch.squeeze(im_x.permute(1, 2, 3, 0))

    return x_ata

# ---------------------------------------------------------------------------
# Compatibility aliases
# Provide common aliases to keep interface consistent with other NUFFT implementations:
# ---------------------------------------------------------------------------

init_multi_op = init_nufft_multi_op
forward_multi_op = nufft_forward_multi_op
adjoint_multi_op = nufft_adjoint_multi_op
ata_multi_op = nufft_toep_ata_multi_op
