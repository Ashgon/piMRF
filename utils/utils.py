"""
Created on Thu Dec 23 2025
@author: Chaoguang Gong
This module provides multiple utility functions for piMRF reconstruction.
"""

import os
import sys
import math
import torch
import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

# Dynamically add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs.piMRF_config as cfg
import utils.Nufft_multi as Nufft_multi
import utils.SIM_EPG as SIM_EPG

# Data types
dtype_float = cfg.dtype_float
dtype_complex = cfg.dtype_complex
dtype_int = cfg.dtype_int
device = cfg.device

def kSVD_get_compressed_ksp_tensor(ksp_und, Vk_in=None, Nufft_multi_handle=Nufft_multi):
    """
    Perform SVD compression on k-space data and return compressed k-space tensor (torch.Tensor version).

    Parameters:
    - ksp_und: undersampled k-space data, shape [num_time, num_coils, num_sample].
    - cfg: configuration object providing related parameters.

    Returns:
    - ksp_tk: compressed k-space data, shape [rank, num_coils, num_interleafs * num_sample].
    - Vk: SVD compression matrix, shape [num_sample, rank].
    """
    # Check input type
    if isinstance(ksp_und, np.ndarray):
        # If input is a NumPy array, convert to PyTorch tensor
        ksp_und = torch.from_numpy(ksp_und).to(dtype=cfg.dtype_complex, device=cfg.device)
    elif not isinstance(ksp_und, torch.Tensor):
        raise ValueError("Input data must be a NumPy array or a PyTorch tensor")

    # Determine Vk (SVD basis) if not provided
    if Vk_in is None:
        # Load dictionary data
        Dic_ori = cfg.dic_m  # [18145, 500]

        # Normalize dictionary
        Dic_normed, norms_Dic = normlize_data(Dic_ori)  # [18145, 500]

        # Compute SVD
        U, S, V_star = scipy.linalg.svd(Dic_normed.detach().cpu().numpy(), full_matrices=False, lapack_driver='gesvd')
        V_star = torch.as_tensor(V_star, dtype=cfg.dtype_complex, device=cfg.device)
        V = V_star.conj().T  # [500, 500]
        Vk = V[:, :cfg.rank]  # [500, rank]
        Vk = Vk.to(dtype=cfg.dtype_complex, device=cfg.device)
    else:
        Vk = Vk_in[:, :cfg.rank]  # [500, rank]

    # Arrange k-space data into full sampling non-grid positions
    # Resulting shape: [ntime, ncoils, nsamples * ninterleaves], with zeros where samples are missing
    time_dim, coil_dim, sample_dim = ksp_und.shape
    ksp_expand_tL = torch.zeros((time_dim, coil_dim, cfg.num_interleafs * cfg.num_samples), dtype=ksp_und.dtype, device=cfg.device)
    exp_index = Nufft_multi_handle.get_batch_index_tensor(cfg.num_interleafs * cfg.num_samples, sample_dim)  # compute fill indices
    for t in range(time_dim):
        start_idx = t % cfg.num_interleafs
        ksp_expand_tL[t, :, exp_index[start_idx][0]:exp_index[start_idx][1]] = ksp_und[t, :, :]
    ksp_expand_tL_2D = ksp_expand_tL.reshape(time_dim, -1)  # (ntime, ncoils * nsamples * ninterleaves)
    # Perform temporal SVD compression
    ksp_expand_tk_2D = torch.matmul(Vk.T, ksp_expand_tL_2D)  # (k, ncoils * nsamples * ninterleaves)
    ksp_expand_tk = ksp_expand_tk_2D.reshape(cfg.rank, coil_dim, cfg.num_interleafs * cfg.num_samples)  # (k, ncoils, ninterleaves*nsamples)
    ktraj_sorted_indices = Nufft_multi_handle.ktraj_full_sorted_index # indices of traj sorted by magnitude
    ktraj_sorted_indices = torch.as_tensor(np.squeeze(ktraj_sorted_indices - 1), dtype=torch.long, device=cfg.device)
    ksp_tk = ksp_expand_tk[:, :, ktraj_sorted_indices]  # reorder according to trajectory
    del ksp_und, ksp_expand_tL, ksp_expand_tL_2D, ksp_expand_tk_2D, ktraj_sorted_indices, ksp_expand_tk
    
    return ksp_tk, Vk

def normlize_data(data):
    '''
    Row-wise normalization (supports complex numbers).

    Inputs:
    - data: complex data matrix [N, L], can be NumPy array or PyTorch tensor.

    Returns:
    - data: row-normalized data matrix [N, L]
    - norms_D: per-row norms [N]
    '''
    # Check input type
    is_numpy = isinstance(data, np.ndarray)
    is_torch = isinstance(data, torch.Tensor)

    if not (is_numpy or is_torch):
        raise ValueError("Input data must be a NumPy array or a PyTorch tensor")

    # Compute L2 norm per row (supports complex values)
    if is_numpy:
        norms_D = np.linalg.norm(data, axis=1, keepdims=True)  # N*1
        norms_D[norms_D == 0] = 1  # Handle zero norms
        data = data / norms_D  # Row-wise normalization N*L
        norms_D = norms_D.flatten()  # Flatten norms to [N,]
    elif is_torch:
        norms_D = torch.linalg.norm(data, ord=2, dim=1, keepdim=True)  # N*1
        norms_D[norms_D == 0] = 1  # Handle zero norms
        data = data / norms_D  # Row-wise normalization N*L
        norms_D = norms_D.squeeze()  # Flatten norms to [N,]

    return data, norms_D

def resize_image(image_input, target_size=cfg.brain_size, mode='nearest', align_corners=False):
    """
    Resample a 2D or multi-channel image (upsample or downsample).

    Supported input types/shapes:
    - torch.Tensor or numpy.ndarray
    - (H, W), (C, H, W), (N, C, H, W)

    Parameters:
    - image_input: input image (torch.Tensor or numpy.ndarray)
    - target_size: (target_height, target_width)
    - mode: interpolation mode, one of 'nearest', 'bilinear', or 'bicubic'
    - align_corners: passed to torch.nn.functional.interpolate when using bilinear/bicubic

    Returns:
    - resampled image with the same type (torch or numpy) and similar dimensional layout as the input

    Notes:
    - For complex tensors, real and imaginary parts are interpolated separately and then recombined.
    - Supports 4D tensors with any batch size (N, C, H, W).
    """
    import torch.nn.functional as F

    was_numpy = isinstance(image_input, np.ndarray)
    np_input = None
    if was_numpy:
        np_input = image_input
        image_input = torch.from_numpy(image_input)

    if not isinstance(image_input, torch.Tensor):
        raise ValueError("image_input must be torch.Tensor or numpy.ndarray")

    orig_dim = image_input.dim()

    # Standardize to (N, C, H, W)
    if orig_dim == 4:
        tensor = image_input
    elif orig_dim == 3:
        tensor = image_input.unsqueeze(0)  # [1, C, H, W]
    elif orig_dim == 2:
        tensor = image_input.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        raise ValueError("Unsupported input dimensions. Supported dimensions are (H,W), (C,H,W), or (N,C,H,W)")

    # Parse target size
    try:
        target_height, target_width = int(target_size[0]), int(target_size[1])
    except Exception:
        raise ValueError("target_size must be an iterable of (height, width)")
    _, _, current_h, current_w = tensor.shape
    # If the target size is the same as the current size, return the original input (preserving type)
    if target_height == current_h and target_width == current_w:
        if was_numpy:
            return np_input
        return image_input

    # Handle complex numbers: interpolate real and imaginary parts separately
    is_complex = torch.is_complex(tensor)

    # Ensure floating point type for interpolation
    float_type = torch.float32
    device = tensor.device

    size = (target_height, target_width)

    if is_complex:
        real = tensor.real.to(dtype=float_type)
        imag = tensor.imag.to(dtype=float_type)
        if mode in ('bilinear', 'bicubic'):
            real_r = F.interpolate(real, size=size, mode=mode, align_corners=align_corners)
            imag_r = F.interpolate(imag, size=size, mode=mode, align_corners=align_corners)
        else:
            real_r = F.interpolate(real, size=size, mode=mode)
            imag_r = F.interpolate(imag, size=size, mode=mode)
        out = torch.complex(real_r, imag_r).to(device=device)
    else:
        tensor_f = tensor.to(dtype=float_type)
        if mode in ('bilinear', 'bicubic'):
            out_f = F.interpolate(tensor_f, size=size, mode=mode, align_corners=align_corners)
        else:
            out_f = F.interpolate(tensor_f, size=size, mode=mode)
        # Restore original data type
        orig_dtype = image_input.dtype
        if orig_dtype.is_floating_point:
            out = out_f.to(dtype=orig_dtype)
        else:
            out = out_f

    # Restore original dimensions
    if orig_dim == 2:
        result = out.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        result = out.squeeze(0)
    else:
        result = out

    # If the original input was numpy, return numpy
    if was_numpy:
        return result.detach().cpu().numpy()

    return result

def build_coordinate_train(L_PE, L_RO):
    x = np.linspace(-1, 1, L_PE)
    y = np.linspace(-1, 1, L_RO)
    (x, y) = np.meshgrid(x, y, indexing='ij')
    xy = np.stack([
        x,
        y], -1).reshape(-1, 2)
    xy = xy.reshape(L_PE, L_RO, 2)
    return xy

def normalize_to_signUnit(T1_tensor, T2_tensor, PD_tensor):
    """
    Normalize the input T1_tensor, T2_tensor, PD_tensor to the range [-1, 1] according to predefined min and max values.
    Parameters:
    - T1_tensor: T1 parameter values
    - T2_tensor: T2 parameter values
    - PD_tensor: PD parameter values
    Returns:
    - T1_normalized: Normalized T1 values
    - T2_normalized: Normalized T2 values
    - PD_normalized: Normalized PD values
    """
    # Automatically get the minimum and maximum values for each channel
    min_T1, max_T1 = cfg.T1_min, cfg.T1_max
    min_T2, max_T2 = cfg.T2_min, cfg.T2_max
    min_PD, max_PD = cfg.PD_min, cfg.PD_max
    
    # Map the values of each channel to [-1, 1]
    T1_normalized = 2 * (T1_tensor - min_T1) / (max_T1 - min_T1) - 1
    T2_normalized = 2 * (T2_tensor - min_T2) / (max_T2 - min_T2) - 1
    PD_normalized = 2 * (PD_tensor - min_PD) / (max_PD - min_PD) - 1

    return T1_normalized, T2_normalized, PD_normalized

def denormalize_to_oriRange(T1_normalized, T2_normalized, PD_normalized=None):
    """
    Map the normalized values of T1, T2, PD channels from [-1, 1] back to their original numerical ranges.

    Parameters:
    - T1_normalized (torch.Tensor or np.ndarray): Normalized T1 values
    - T2_normalized (torch.Tensor or np.ndarray): Normalized T2 values
    - PD_normalized (torch.Tensor or np.ndarray, optional): Normalized PD values

    Returns:
    - tuple: T1, T2, PD values mapped back to their original ranges (if PD is provided)
    """
    def denormalize(normalized, min_val, max_val):
        """General denormalization function"""
        return (normalized + 1) * (max_val - min_val) / 2 + min_val

    results = [denormalize(T1_normalized, cfg.T1_min, cfg.T1_max),
               denormalize(T2_normalized, cfg.T2_min, cfg.T2_max)]

    if PD_normalized is not None:
        results.append(denormalize(PD_normalized, cfg.PD_min, cfg.PD_max))

    return tuple(results)

def denormalize_single_channel(normalized_tensor, min_val, max_val):
    """
    Denormalize a single-channel normalized tensor to its original range.
    
    Parameters:
    - normalized_tensor: Normalized tensor
    - min_val: Minimum value of the original range
    - max_val: Maximum value of the original range
    
    Returns:
    - denormalized_tensor: Denormalized tensor
    """
    # Inverse normalization operation, mapping standardized values back to the original range
    denormalized_tensor = (normalized_tensor + 1) / 2 * (max_val - min_val) + min_val
    return denormalized_tensor

def psnr(recon, ref):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between reconstructed and reference images.

    Parameters:
    recon: Reconstructed image.
    ref: Reference image.

    Returns:
    float: PSNR value between reconstructed and reference images (in decibels).
    """
    # recon: reconstructed image, ref: reference image
    
    # Check if recon and ref are tensor variables, convert to numpy if so
    if isinstance(recon, torch.Tensor):
        recon = recon.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()

    scale = np.max(np.abs(ref))

    recon_data = recon/scale
    ref_data = ref/scale
    mse = np.mean( np.abs(recon_data - ref_data) ** 2 )
    psnr = 10 * math.log10(1.0/mse)

    return psnr

def calculate_nmse(ref, x):
    # Calculate mean squared error between original and predicted images

    # Check if recon and ref are tensor variables, convert to numpy if so
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    mse_value = np.mean((ref - x) ** 2)

    # Calculate variance of reference image
    ref_var = np.var(ref.flatten())

    # Calculate normalized mean squared error
    nmse = mse_value / ref_var
    return nmse

def AWGN(x, snr):
    """
    Additive White Gaussian Noise
    :param x: Raw Signal
    :param snr: SNR (dB)
    :return: Received Signal With AWGN
    """
    # Set random seed
    np.random.seed(cfg.seed_value)
    # Calculate signal power
    signal_power = np.mean(np.abs(x) ** 2)
    
    # Convert SNR from dB to linear value
    snr_linear = 10 ** (snr / 10)
    
    # Calculate noise power
    noise_power = signal_power / snr_linear  

    if np.iscomplexobj(x):
        # Generate Gaussian white noise (real and imaginary parts)
        noise_real = np.sqrt(noise_power / 2) * np.random.randn(*x.shape).astype(x.dtype)
        noise_imag = np.sqrt(noise_power / 2) * np.random.randn(*x.shape).astype(x.dtype)
        
        # Generate complex noise
        noise = noise_real + 1j * noise_imag        
    else:
        # Generate Gaussian white noise (real part only)
        noise = np.sqrt(noise_power) * np.random.randn(*x.shape).astype(x.dtype)
    
    # Add noise to signal
    noisy_signal = x + noise.astype(x.dtype)
    return noisy_signal

def calculate_snr(x, x_noisy):
    """
    Calculate the SNR of signal after adding noise, test if noise addition is correct
    :param x: Original signal
    :param x_noisy: Signal with added noise
    :return: SNR (dB)
    """
    # Calculate power of original signal
    signal_power = np.mean(np.abs(x) ** 2)
    
    # Calculate noise signal
    noise = x_noisy - x
    
    # Calculate noise power
    noise_power = np.mean(np.abs(noise) ** 2)
    
    # Calculate SNR (linear value)
    snr_linear = signal_power / noise_power
    
    # Convert SNR to dB
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db

def show_tissue_maps(pre_T1, pre_T2, pre_PD, tissue_mask, OUT_dir, iter, losses=None):
    '''
    Visualize maps_toshow=[pre_T1, pre_T2, pre_PD, tissue_mask], loss curves
    '''
    plt.ioff()
    plt.figure(figsize=(90, 16))  # Adjust size to fit five subplots
    T1_show = plt.subplot(151)
    T2_show = plt.subplot(152)
    PD_show = plt.subplot(153)
    Mask_show = plt.subplot(154)  # Plot mask
    Loss_show = plt.subplot(155)  # Plot loss curves

    im_T1 = T1_show.pcolor(pre_T1.detach().cpu().numpy(), cmap=cfg.colormap)
    T1_show.set_title('T1')
    T1_show.axis('off')  # Hide coordinate axes
    plt.colorbar(im_T1, ax=T1_show, orientation='vertical')
    
    im_T2 = T2_show.pcolor(pre_T2.detach().cpu().numpy(), cmap=cfg.colormap)
    T2_show.set_title('T2')
    T2_show.axis('off')  # Hide coordinate axes
    plt.colorbar(im_T2, ax=T2_show, orientation='vertical')

    im_PD = PD_show.pcolor(pre_PD.detach().cpu().numpy(), cmap=cfg.colormap)
    PD_show.set_title('PD')
    PD_show.axis('off')  # Hide coordinate axes
    plt.colorbar(im_PD, ax=PD_show, orientation='vertical')

    # Plot mask tissue_mask
    im_Mask = Mask_show.pcolor(tissue_mask.detach().cpu().numpy(), cmap='gray')
    Mask_show.set_title('Tissue Mask')
    Mask_show.axis('off')  # Hide coordinate axes
    plt.colorbar(im_Mask, ax=Mask_show, orientation='vertical')

    # Loss curve display
    if losses is not None and len(losses) > 0:
        # Extract dc_data_loss, dc_csm_loss and tv_loss        
        dc_losses = [loss[1] for loss in losses] if len(losses[0]) > 1 else None
        tv_losses = [loss[2] for loss in losses] if len(losses[0]) > 2 else None
        dc_csm_losses = [loss[3] for loss in losses] if len(losses[0]) > 3 else None

        # Plot DC_data loss and DC_csm loss
        Loss_show.plot(dc_losses, label='DC_data Loss', color='b')
        Loss_show.text(len(dc_losses) - 1, dc_losses[-1], f'{dc_losses[-1]:.2f}', 
               color='b', va='bottom', ha='right')
        if dc_csm_losses is not None:
            Loss_show.plot(dc_csm_losses, label='DC_csm Loss', color='g')
            Loss_show.text(len(dc_csm_losses) - 1, dc_csm_losses[-1], f'{dc_csm_losses[-1]:.2f}', 
                   color='g', va='bottom', ha='right')
        ax2 = Loss_show.twinx()  # Create a second y-axis sharing the x-axis
        # Create dual y-axes
        if tv_losses is not None:            
            ax2.plot(tv_losses, label='TV Loss', color='r')
            ax2.text(len(tv_losses) - 1, tv_losses[-1], f'{tv_losses[-1]:.2f}', 
             color='r', va='bottom', ha='right')
        # Set y-axis labels
        ax2.set_ylabel('TV Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Set title, labels and legend
        Loss_show.set_title('Loss Curves')
        Loss_show.set_xlabel('Iteration')
        Loss_show.set_ylabel('DC_data and DC_csm Loss', color='b')
        Loss_show.tick_params(axis='y', labelcolor='b')
        Loss_show.legend(loc='upper left')

        # Add TV loss legend
        if tv_losses is not None:
            ax2.legend(loc='upper right')        

    # Automatically adjust subplot layout
    plt.tight_layout()

    plt.savefig(OUT_dir+'/A_tissueMaps_{}.jpg'.format(iter))
    plt.close()

def show_sensitivity_maps(tstCsm_tensor, OUT_dir, iter):
    '''
    Plot magnitude and phase maps of all coil sensitivity maps and display them in one image
    Magnitude subplots use a unified colorbar, phase subplots also use a unified colorbar
    Save output to specified folder
    '''
    with torch.no_grad():
        tstCsm_tensor = tstCsm_tensor.squeeze()  # [ncoils, Nx, Ny]
        
        num_maps = tstCsm_tensor.shape[0]  # ncoils
        
        # Calculate appropriate subplot grid size
        cols = 8
        rows = (num_maps + cols - 1) // cols
        
        plt.ioff()
        fig, axes = plt.subplots(2 * rows, cols, figsize=(cols * 3, rows * 3))  # Adjust size to reduce save time
        
        # Calculate magnitude and phase of all sensitivity maps and convert to NumPy arrays
        magnitudes = torch.abs(tstCsm_tensor).detach().cpu().numpy()
        phases = np.where(magnitudes > 0, torch.angle(tstCsm_tensor).detach().cpu().numpy(), 0)

        # Calculate global range of magnitude and phase
        mag_min, mag_max = magnitudes.min(), magnitudes.max()
        phase_min, phase_max = phases.min(), phases.max()

        for i in range(num_maps):
            # Determine current subplot position
            row, col = divmod(i, cols)
            
            # Display magnitude map
            ax_mag = axes[2 * row, col]
            im_mag = ax_mag.imshow(magnitudes[i], cmap='jet', aspect='auto', vmin=mag_min, vmax=mag_max)
            ax_mag.set_title(f'SensMap {i+1} Magnitude', fontsize=8)
            ax_mag.axis('off')
            
            # Display phase map
            ax_phase = axes[2 * row + 1, col]
            im_phase = ax_phase.imshow(phases[i], cmap='hsv', aspect='auto', vmin=phase_min, vmax=phase_max)
            ax_phase.set_title(f'SensMap {i+1} Phase', fontsize=8)
            ax_phase.axis('off')
        
        # Add unified colorbar
        cbar_ax_mag = fig.add_axes([0.92, 0.55, 0.015, 0.35])  # Magnitude colorbar
        cbar_ax_phase = fig.add_axes([0.92, 0.1, 0.015, 0.35])  # Phase colorbar
        fig.colorbar(im_mag, cax=cbar_ax_mag, orientation='vertical')
        fig.colorbar(im_phase, cax=cbar_ax_phase, orientation='vertical')

        # Handle extra subplots (if any)
        for j in range(num_maps, rows * cols):
            fig.delaxes(axes.flatten()[2 * j])  # Delete extra magnitude subplot
            fig.delaxes(axes.flatten()[2 * j + 1])  # Delete extra phase subplot
    
        fig.subplots_adjust(right=0.9)  # Adjust subplot layout to accommodate manually added colorbar
        
        # Save image to specified directory
        plt.savefig(f'{OUT_dir}/SensMaps_{iter}.jpg', dpi=150)  # Lower DPI to speed up saving
        plt.close()

def process_trajectory(traj, para_length=None, n_interleaves=None, verbose=False):
    """
    General trajectory processing function supporting input:
      1) Complex k-space trajectory, shape (Nreads, T)

    Rules:
    - Denote time dimension as T, single period length as n_interleaves, final time length as para_length.
    - For complex trajectory input:
        * If T == n_interleaves: check max magnitude max_abs; if max_abs > np.pi, scale the entire trajectory to max_abs==np.pi (preserving phase),
          then repeat (periodic extension) in the time dimension until length is at least para_length, then truncate to para_length.
        * If T > n_interleaves: if T >= para_length, truncate to para_length and check/normalize magnitude; otherwise repeat/extend existing trajectory in time to para_length and check magnitude.
        * If T < n_interleaves: raise ValueError (because trajectory period information is incomplete).    

    Parameters:
      traj: np.ndarray or torch.Tensor, supports complex or real
      para_length: Target time length, defaults to cfg.para_length
      n_interleaves: Single period length, defaults to cfg.n_interleaves
      verbose: Whether to print processing information

    Returns:
      Processed trajectory of the same type as input (time dimension is para_length)
    """
    if para_length is None:
        para_length = cfg.para_length
    if n_interleaves is None:
        n_interleaves = cfg.num_interleafs

    is_torch = isinstance(traj, torch.Tensor)
    is_numpy = isinstance(traj, np.ndarray)
    if not (is_torch or is_numpy):
        raise ValueError('traj must be numpy.ndarray or torch.Tensor')

    # helpers
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def to_torch(x, ref):
        if isinstance(ref, torch.Tensor):
            t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            return t.to(device=ref.device, dtype=ref.dtype)
        return x

    # Use numpy view for shape operations and calculations, then convert back to original type
    traj_np = to_numpy(traj)

    # Determine dimensions
    if traj_np.ndim == 2:
        # Complex k-space trajectory (Nreads, T)
        Nreads, T = traj_np.shape
        if verbose:
            print(f'process_trajectory: detected complex k-space traj shape {traj_np.shape}')

        if T < n_interleaves:
            raise ValueError(f"Time dimension T={T} is smaller than single period n_interleaves={n_interleaves}")

        # Define magnitude check/normalization function
        def check_and_normalize(mat):
            max_abs = np.max(np.abs(mat))
            if verbose:
                print(f'process_trajectory: max_abs={max_abs}')
            if max_abs > np.pi:
                scale = np.pi / max_abs
                if verbose:
                    print(f'process_trajectory: scaling by {scale}')
                return mat * scale
            return mat

        if T == n_interleaves:
            traj_proc = check_and_normalize(traj_np)
            # Repeat to para_length
            reps = int(np.ceil(para_length / T))
            traj_rep = np.tile(traj_proc, (1, reps))
            traj_final = traj_rep[:, :para_length]

        else:  # T > n_interleaves
            if T >= para_length:
                traj_cut = traj_np[:, :para_length]
                traj_final = check_and_normalize(traj_cut)
            else:
                # T < para_length but > n_interleaves: extend
                raise ValueError(f"Time dimension T={T} is larger than n_interleaves={n_interleaves} but smaller than para_length={para_length}, cannot extend meaningfully. Please check the input trajectory or interleaves setting.")

    # Convert back to original type
    if is_torch:
        traj_out = to_torch(traj_final, traj)
    else:
        traj_out = traj_final

    return traj_out


class SIMU_EPG():

    def __init__(self, fft_hook, alpha, TRs, TEs, **kwargs):  # **kwargs: special parameter to accept any number of keyword arguments
        self.fft_hook = fft_hook
        self.FAs = alpha
        self.TRs = TRs
        self.TEs = TEs
        self.kwargs = kwargs
    
    def get_LUT_jiangyun(self):
        '''
        Get lookup table based on scanning parameters provided in jiangyun-paper
        REF: Jiang et al., "MR Fingerprinting Using Fast Imaging with Steady State Precession (FISP) with Spiral Readout: MR Fingerprinting with FISP," Magn. Reson. Med., vol. 74, no. 6, pp. 1621–1631, Dec. 2015.
        T1:
        range: 100~3000, step:10
        range: 3000~5000, step:200
        T2:
        range: 10~300, step:5
        range: 300~500, step:50
        Output:
        LUT: Entries*2 [T1, T2]
        '''
        T1 = np.float64(list(range(100, 3001, 10))+list(range(3200, 5001, 200)))
        T2 = np.float64(list(range(10, 301, 5))+list(range(350, 501, 50)))
        LUT = np.zeros((len(T1)*len(T2), 2), dtype=np.float64)
        k = 0
        for tmp_T1 in T1:
            for tmp_T2 in T2:
                if tmp_T1 < tmp_T2:  # Tissue T1 values are greater than T2 values with a difference of 5-10x, here we exclude cases where T1 < T2
                    continue
                LUT[k, 0] = tmp_T1
                LUT[k, 1] = tmp_T2
                k = k+1
        # Remove excess entries that are all zeros without curves
        LUT = LUT[0:k, :]
        return torch.from_numpy(LUT).type(dtype_float).to(device)
    
    def Dic_gen(self):
        '''
        Bloch simulation, generate dictionary, sequence IR_FISP
        Input:
        FAs: Flip angle sequence in degrees
        TE: Echo Time in ms
        TRs: Pulse sequence repetition period sequence in ms 
        Other notes:
        M0: Steady-state magnetization (proton density) (default value is 1 when generating dictionary)
        T1: Longitudinal relaxation time, dictionary specified range in ms
        T2: Transverse relaxation time, dictionary specified range in ms
        Output:
        D：A dictionary containing all signal entries [Entries*L]
        LUT: A lookup table corresponding to the dictionary [Entries*2]
        '''
        LEN = self.FAs.shape[0]
        LUT = self.get_LUT_jiangyun()  # [Entries*2]
        FA_init = torch.from_numpy(self.FAs.astype(np.float64).reshape([LEN, 1]))
        TR_init = torch.from_numpy(self.TRs.astype(np.float64).reshape([LEN, 1]))
        FAs_TRs = torch.cat([FA_init, TR_init], 1).type(dtype_float).to(device)
        FAs_TRs = FAs_TRs.view([1, LEN, 2])
        TEs_init = torch.from_numpy(self.TEs.astype(np.float64).reshape([LEN, 1])).type(dtype_float).to(device)
        dictionary = SIM_EPG.epg_ir_fisp_signal_batch(
            FAs_TRs, TEs_init, LUT[:, 0], LUT[:, 1])
        return dictionary.detach().cpu().numpy(), LUT.detach().cpu().numpy()
    
    def KspaceUnd(self, T1, T2, PD, Brain_mask=None, Pre_csm=None):
        '''
        Perform forward physical simulation based on the given parameter maps T1, T2, PD to generate undersampled k-space data.
        Inputs:
        - T1: T1 parameter map
        - T2: T2 parameter map
        - PD: PD parameter map
        - Brain_mask: Brain tissue mask
        - FA: Sequence of flip angles
        - TR: Sequence of repetition times
        - TE: Sequence of echo times
        - Pre_csm: Predicted susceptibility map

        Outputs:
        - img_signal: Simulated image signal [L, 1, N_samples]
        - ksp_und: Undersampled k-space data [L, 1, N_samples]
        '''    
        xvoxels, yvoxels = T1.shape[:2]
        T1 = T1 * Brain_mask[:, :, 0]
        T2 = T2 * Brain_mask[:, :, 1]
        PD = PD * Brain_mask[:, :, 2]
        N = self.FAs.shape[0] - 1
 
        T1_flatten = T1.flatten()
        T2_flatten = T2.flatten()
        M0_flatten = PD.flatten()
        non_zero_indices = torch.nonzero((T1_flatten != 0) & (T2_flatten != 0) & (M0_flatten != 0) & (T1_flatten > T2_flatten))

        non_zero_T1values = T1_flatten[non_zero_indices].squeeze()
        non_zero_T2values = T2_flatten[non_zero_indices].squeeze()
        non_zero_M0values = M0_flatten[non_zero_indices].squeeze()
        del T1_flatten, T2_flatten, M0_flatten
        non_zero_signal = torch.zeros((non_zero_T1values.shape[0], N), dtype=cfg.dtype_complex, device=cfg.device)
        non_zero_signal = SIM_EPG.build_TemplateMatrix_mat(self.FAs, self.TRs, self.TEs,
                                                            non_zero_T1values.detach().cpu().numpy(), 
                                                            non_zero_T2values.detach().cpu().numpy(), 
                                                            non_zero_M0values.detach().cpu().numpy())
        np_non_zero_indices = non_zero_indices[:, 0]
        signal = torch.zeros((xvoxels*yvoxels, N), dtype=cfg.dtype_complex, device=cfg.device)
        signal[np_non_zero_indices] = non_zero_signal
        img_signal = signal.reshape(xvoxels, yvoxels, N)
        
        # perform undersampling
        if Pre_csm is None:
            ksp_und = self.fft_hook.forward_multi_op(img_signal, smap=self.fft_hook.csm, ktraj=self.fft_hook.ktraj_und)
        else:
            ksp_und = self.fft_hook.forward_multi_op(img_signal, smap=Pre_csm, ktraj=self.fft_hook.ktraj_und)

        return img_signal.detach().cpu().numpy(), ksp_und.detach().cpu().numpy()
