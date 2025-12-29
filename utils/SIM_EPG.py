'''
Created on Jun 12 2024
@author: Peng Li (https://github.com/bigponglee)
This module implements the Extended Phase Graph (EPG) simulation for MRI signal generation.
'''

import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import configs.piMRF_config as cfg


# Data types
dtype_float = cfg.dtype_float
dtype_complex = cfg.dtype_complex
device = cfg.device


N_states = 20  # Number of states to simulate
phi = 90.  # degrees, this function assumes phi = 90 for all real states, but can be any number
# create a matrix that is size 3 x N_states, for shifting without wrap, put ones in off diagonals
_mask_F_plus = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_mask_F_plus[0, :] = 1.0
_mask_F_minus = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_mask_F_minus[1, :] = 1.
_mask_Z = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_mask_Z[2, :] = 1.
_F0_plus_mask = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_F0_plus_mask[0, 0] = 1.
_shift_right_mat = torch.roll(
    torch.eye(N_states, dtype=dtype_complex), 1, 1).to(device)
_shift_right_mat[:, 0] = 0
_shift_left_mat = torch.roll(
    torch.eye(N_states, dtype=dtype_complex), -1, 1).to(device)
_shift_left_mat[:, -1] = 0


def get_F_plus_states(FZ):
    return _mask_F_plus.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def get_F_minus_states(FZ):
    return _mask_F_minus.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def get_Z_states(FZ):
    # elementwise multiplication
    return _mask_Z.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def get_F0_plus(FZ):
    return _F0_plus_mask.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def grad_FZ(FZ):
    # shift the states
    out1 = torch.matmul(get_F_plus_states(FZ), _shift_right_mat.view([1, N_states, N_states]).expand(FZ.shape[0], N_states, N_states)) + \
        torch.matmul(get_F_minus_states(FZ), _shift_left_mat.view(
            [1, N_states, N_states]).expand(FZ.shape[0], N_states, N_states)) + get_Z_states(FZ)
    # fill in F0+ using a mask
    out2 = out1 + get_F0_plus(torch.conj(torch.roll(out1, -1, 1)))
    return out2


def rf_epg(alpha, phi):
    '''
    Bloch simulation of RF pulse activation, rotation matrix R calculation
    Input:
    alpha: Flip angle (in radians).
    phi = angle of rotation axis from Mz (radians).
    Output:
    R = RF rotation matrix (3x3).
    '''
    phi = torch.as_tensor(phi, dtype=dtype_float)
    R = torch.zeros(alpha.shape[0], 3, 3, dtype=dtype_complex).to(device)
    R[:, 0, 0] = torch.pow(torch.cos(alpha/2.0), 2)
    R[:, 0, 1] = torch.exp(2*1j*phi)*torch.pow(torch.sin(alpha/2.0), 2)
    R[:, 0, 2] = -1j*torch.exp(1j*phi)*torch.sin(alpha)
    R[:, 1, 0] = torch.exp(-2j*phi)*torch.pow(torch.sin(alpha/2.0), 2)
    R[:, 1, 1] = torch.pow(torch.cos(alpha/2.0), 2)
    R[:, 1, 2] = 1j*torch.exp(-1j*phi)*torch.sin(alpha)
    R[:, 2, 0] = -1j/2.0*torch.exp(-1j*phi)*torch.sin(alpha)
    R[:, 2, 1] = 1j/2.0*torch.exp(1j*phi)*torch.sin(alpha)
    R[:, 2, 2] = torch.cos(alpha)
    return R


def rf_FZ(FZ, alpha, phi):
    '''
    Bloch simulation of initial inversion pulse activation
    Input:
    FZ: 3xN vector of F+, F- and Z states.
    alpha: Flip angle (in degrees).
    phi = angle of rotation axis from Mz (in degrees).
    Output:
    Updated FpFmZ state.
    '''
    R = rf_epg(alpha*torch.pi/180., phi*torch.pi/180.)
    return torch.matmul(R, FZ)


def relax_epg(M0, T, T1, T2):
    '''
    Bloch simulation of magnetization relaxation
    Input:
    M0: Steady-state magnetization / magnetization before relaxation
    T: Relaxation time
    T1: Longitudinal relaxation
    T2: Transverse relaxation
    Output:
    A: spin relaxation
    B: input matrix
    '''
    E1 = torch.exp(-T/T1)
    E2 = torch.exp(-T/T2)
    # decay of states due to relaxation
    A = torch.zeros(T1.shape[0], 3, 3, dtype=dtype_complex).to(device)
    A[:, 0, 0] = E2
    A[:, 1, 1] = E2
    A[:, 2, 2] = E1
    B = torch.zeros(T1.shape[0], 3, N_states, dtype=dtype_complex).to(device)
    B[:, 2, 0] = M0*(1.0-E1)
    return A, B


def relax_FZ(FZ, M0, T, T1, T2):
    '''
    Bloch simulation of magnetization relaxation after initial inversion pulse
    Input:
    FZ: Initial magnetization matrix
    M0: Steady-state magnetization / magnetization before relaxation
    T: Relaxation time
    T1: Longitudinal relaxation
    T2: Transverse relaxation
    Output:
    Magnetization matrix after relaxation
    '''
    A, B = relax_epg(M0, T, T1, T2)
    return torch.matmul(A, FZ)+B


def init_ir_relax(M0, inversion_delay, T1, T2):
    '''
    Relaxation after 180° inversion pulse application
    Input:
    M0: Steady-state magnetization (proton density)
    inversion_delay: Recovery time after 180° inversion pulse application
    T1: Longitudinal relaxation time
    T2: Transverse relaxation time
    Output:
    Initial signal state before next pulse application
    '''
    m1 = torch.as_tensor([[0.], [0.], [M0]], dtype=dtype_complex).to(
        device).view([1, 3, 1]).expand(T1.shape[0], 3, 1)
    m2 = torch.zeros(T1.shape[0], 3, N_states-1,
                     dtype=dtype_complex).to(device)
    FZ_init = torch.cat((m1, m2), 2)
    init_FA = torch.as_tensor(
        np.ones(T1.shape[0]) * 180.0, dtype=dtype_float).to(device)
    FZ_flip = relax_FZ(rf_FZ(FZ_init, init_FA, phi),
                       M0, inversion_delay, T1, T2)
    return FZ_flip


def get_rf_epg(FZ, FA):
    '''
    Simulate RF pulse activation
    Input:
    FZ: 3xN vector of F+, F- and Z states.
    FA: Flip angle (°)
    Output:
    Updated FpFmZ state.
    '''
    R = rf_epg(FA*torch.pi/180.0, phi*torch.pi/180.0)
    return torch.matmul(R, FZ)


def get_relax_epg(FZ, M0, T, T1, T2):
    '''
    Signal generated after current pulse excitation
    Input:
    M0: Steady-state magnetization (proton density)
    T: Recovery time after pulse application
    T1: Longitudinal relaxation time
    T2: Transverse relaxation time
    Output:
    Pulse excitation signal
    '''
    FZ_out = relax_FZ(FZ, M0, T, T1, T2)
    return FZ_out


def next_rf_relax(FZ, M0, T, T1, T2):
    '''
    Initial signal state before next pulse
    Input:
    M0: Steady-state magnetization (proton density)
    T: Recovery time
    T1: Longitudinal relaxation time
    T2: Transverse relaxation time
    Output:
    Signal before next pulse excitation
    '''
    FZ_spoiled = grad_FZ(FZ)
    return get_relax_epg(FZ_spoiled, M0, T, T1, T2)


def epg_ir_fisp_signal_batch(FAs_TRs, TEs, T1, T2):
    '''
    Bloch simulation, generate signal, sequence IR_FISP
    Input:
    M0: Steady-state magnetization (proton density)
    FAs: Flip angle in degrees
    TEs: Echo Time in ms
    TRs: Pulse sequence repetition period in ms
    inversion_delay: Recovery time after 180° inversion pulse application in ms
    T1: Longitudinal relaxation time in ms
    T2: Transverse relaxation time in ms
    Output:
    signal: Response signal
    '''
    M0 = torch.as_tensor(1.0).to(device)
    LEN = FAs_TRs.shape[1]
    signal = torch.zeros(T1.shape[0], LEN, dtype=dtype_complex).to(device)
    W = torch.zeros(T1.shape[0], 3, N_states, dtype=dtype_complex).to(device)
    W[:, 2, 0] = 1.0
    for i in range(LEN):
        U = get_rf_epg(
            W, FAs_TRs[:, i, 0])
        V = get_relax_epg(U, M0, TEs[i], T1, T2)
        W = next_rf_relax(
            V, M0, FAs_TRs[:, i, 1]-TEs[i], T1, T2)
        signal[:, i] = V[:, 0, 0]
    return signal[:, 1:]


def build_TemplateMatrix(FAs, TRs, TEs, T1_flat, T2_flat, M0_flat):
    '''
    Bloch simulation, generate time-domain matrix X, sequence IR_FISP
    Input:
    FAs: Flip angle sequence in degrees
    TEs: Echo Time in ms
    TRs: Pulse sequence repetition period sequence in ms
    inversion_delay: Recovery time after 180° inversion pulse application in ms
    M0_maps: Steady-state magnetization (proton density) (default value is 1 when generating dictionary)
    T1_maps: Longitudinal relaxation time in ms
    T2_maps: Transverse relaxation time in ms
    Output:
    X: Simulated time-domain matrix
    '''
    LEN = FAs.shape[0]
    num_voxels = T1_flat.shape[0]
    LUT = np.zeros([num_voxels, 2])
    LUT[:, 0] = T1_flat
    LUT[:, 1] = T2_flat
    LUT = torch.from_numpy(LUT).type(dtype_float).to(device)
    M0_maps = torch.from_numpy(M0_flat.reshape(
        [num_voxels, 1])).type(dtype_float).to(device)
    FA_init = torch.from_numpy(FAs.astype(np.float64).reshape([LEN, 1]))
    TR_init = torch.from_numpy(TRs.astype(np.float64).reshape([LEN, 1]))
    TE_init = torch.from_numpy(TEs.astype(np.float64).reshape([LEN, 1])).type(dtype_float).to(device)
    FAs_TRs = torch.cat([FA_init, TR_init], 1).type(dtype_float).to(device)
    FAs_TRs = FAs_TRs.view([1, LEN, 2])
    X = epg_ir_fisp_signal_batch(
        FAs_TRs, TE_init, LUT[:, 0], LUT[:, 1])
    X = X*M0_maps
    return X.reshape([num_voxels, LEN-1])


def build_TemplateMatrix_mat(FAs, TRs, TEs, T1_flat, T2_flat, M0_flat):
    '''
    Bloch simulation, generate time-domain matrix X, sequence IR_FISP
    Input:
    FAs: Flip angle sequence in degrees [L, ]
    TRs: Pulse sequence repetition period sequence in ms [L, ]
    M0_maps: Steady-state magnetization (proton density) (default value is 1 when generating dictionary) [Nx,Ny]
    T1_maps: Longitudinal relaxation time in ms [Nx,Ny]
    T2_maps: Transverse relaxation time in ms [Nx,Ny]
    Output:
    X: Simulated time-domain matrix [Nx,Ny, L]
    '''
    X = build_TemplateMatrix(FAs, TRs, TEs, T1_flat, T2_flat, M0_flat)
    return X
