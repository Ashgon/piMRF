"""
Created on Thu Dec 23 2025
@author: Chaoguang Gong
This module defines the model architectures and loss functions for piMRF reconstruction.
REF:
1. DINER: Zhu et al., "Disorder-Invariant Implicit Neural Representation," IEEE TPAMI, vol. 46, no. 8, pp. 5463–5478, Aug. 2024.
2. Feng et al., "IMJENSE: Scan-Specific Implicit Representation for Joint Coil Sensitivity and Image Estimation in Parallel MRI," IEEE Trans. Med. Imaging, vol. 43, no. 4, pp. 1539–1553, Apr. 2024.
"""

import torch
from torch import nn
import numpy as np
import configs.piMRF_config as cfg
import torch.nn.functional as F
import utils.utils as utils

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out


class DinerSiren_skipCon(nn.Module):
    def __init__(self,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length, in_features))*2 -1), requires_grad = True)
                
        self.first_layer = SineLayer(in_features, hidden_features, 
                                     is_first=True, omega_0=first_omega_0)
        self.first_bn = nn.BatchNorm1d(hidden_features)  # First BatchNorm1d layer

        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()  # Store hidden layers using ModuleList
        self.batchnorm_layers = nn.ModuleList()  # Store corresponding BatchNorm1d using ModuleList

        for i in range(hidden_layers):
            self.hidden_layers.append(SineLayer(hidden_features, hidden_features, 
                                                is_first=False, omega_0=hidden_omega_0))
            self.batchnorm_layers.append(nn.BatchNorm1d(hidden_features))  # Corresponding BatchNorm1d

        # Final layer, determine whether to use linear output
        if outermost_linear:
            self.final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                  np.sqrt(6 / hidden_features) / hidden_omega_0)
        else:
            self.final_linear = SineLayer(hidden_features, out_features, 
                                          is_first=False, omega_0=hidden_omega_0)
    
    def forward(self, coords):
        first_output = self.first_layer(self.table)  # Input learnable hash table
        first_output = self.first_bn(first_output)  # First BatchNorm1d

        # Apply skip connections to hidden layers
        x = first_output
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)  # Pass through SineLayer
            x = self.batchnorm_layers[i](x)  # Pass through BatchNorm1d
            x = x * first_output  # Skip connection: element-wise multiplication, applied after BatchNorm1d

        # Final output layer
        output = torch.squeeze(self.final_linear(x))
        # output = torch.clamp(output, min = -1.0,max = 1.0)
        output = torch.tanh(output)
        return output


class MYTVLoss(nn.Module):
    """
    Comprehensive TV Loss supporting multiple modes:
    - 'l1': Standard L1 TV (anisotropic)
    - 'l2': L2 TV (smoother)
    - 'huber': Huber TV (balances sharpness and smoothness) - Recommended
    - 'anisotropic': Anisotropic TV (same as l1)
    - 'isotropic': Isotropic TV sqrt(dx^2 + dy^2)
    - 'charbonnier': Charbonnier/Pseudo-Huber TV (smooth and differentiable)
    - 'lp': Lp-TV (0<p<1 non-convex, stronger edge preservation)
    - 'tgv': Total Generalized Variation (second-order TV, reduces staircase effects)
    - 'weighted': Weighted TV (based on mask or weight map)
    - 'nonlocal': Non-local TV (based on patch similarity)
    - 'directional': Directional/structure tensor TV (preserves structure along directional features)
    """
    def __init__(self, mode='huber', delta=0.01, delta_mode='median', delta_scale=1.0,
                 eps=1e-3, p=0.8, alpha1=1.0, alpha2=0.1, patch_size=5, 
                 search_window=11, sigma=1.0, weight_map=None):
        """
        :param mode: TV type ['l1','l2','huber','anisotropic','isotropic','charbonnier','lp','tgv','weighted','nonlocal','directional']
        :param delta: Huber threshold (when delta_mode='fixed' or as initial value for learnable)
        :param delta_mode: How to determine delta: 'fixed', 'median', 'mean', 'learnable'
        :param delta_scale: Scaling factor for statistical methods
        :param eps: Smoothing parameter for Charbonnier/Isotropic
        :param p: Exponent for Lp-TV (0 < p < 1)
        :param alpha1, alpha2: First-order and second-order weights for TGV
        :param patch_size: Patch size for Non-local TV
        :param search_window: Search window size for Non-local TV
        :param sigma: Standard deviation of Gaussian kernel for Directional TV
        :param weight_map: Weight map for Weighted TV (if None, uses mask)
        """
        super(MYTVLoss, self).__init__()
        self.mode = mode
        self.register_buffer('base_delta', torch.tensor(float(delta)))
        self.delta_mode = delta_mode
        self.delta_scale = float(delta_scale)
        self.eps = eps
        self.p = p
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.patch_size = patch_size
        self.search_window = search_window
        self.sigma = sigma
        self.weight_map = weight_map

        # Learnable delta
        if self.delta_mode == 'learnable':
            self.learnable_delta = nn.Parameter(torch.tensor(float(delta)))
        else:
            self.learnable_delta = None
    
    def _get_delta(self, abs_dx, abs_dy):
        """Compute adaptive or learnable delta value"""
        eps = 1e-6
        if self.delta_mode == 'fixed':
            return float(self.base_delta)
        elif self.delta_mode in ('median', 'mean'):
            combined = torch.cat((abs_dx.flatten(), abs_dy.flatten()))
            if combined.numel() == 0:
                return float(self.base_delta)
            stat = torch.median(combined) if self.delta_mode == 'median' else torch.mean(combined)
            return max(eps, float(stat.item()) * self.delta_scale)
        elif self.delta_mode == 'learnable' and self.learnable_delta is not None:
            return float(torch.abs(self.learnable_delta).item()) + eps
        else:
            return float(self.base_delta)
    
    def _compute_structure_tensor(self, x):
        """Compute structure tensor (for directional TV)"""
        # Compute gradient using Sobel operator
        dx = x[1:, :] - x[:-1, :]
        dy = x[:, 1:] - x[:, :-1]

        # To support 'replicate' padding and conv2d, expand gradients to 4D: [N=1, C=1, H, W]
        dx4 = dx.unsqueeze(0).unsqueeze(0)
        dy4 = dy.unsqueeze(0).unsqueeze(0)

        # Pad dimensions (using replicate pad on 4D)
        # pad format: (pad_left, pad_right, pad_top, pad_bottom)
        dx4 = F.pad(dx4, (0, 0, 0, 1), mode='replicate')
        dy4 = F.pad(dy4, (0, 1, 0, 0), mode='replicate')

        # Structure tensor components (still 4D)
        Jxx4 = dx4 * dx4
        Jyy4 = dy4 * dy4
        Jxy4 = dx4 * dy4

        # Gaussian smoothing (convolution requires 4D input and 4D kernel)
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        gaussian_kernel = self._get_gaussian_kernel(kernel_size, self.sigma, x.device, x.dtype)

        Jxx = F.conv2d(Jxx4, gaussian_kernel, padding=kernel_size // 2).squeeze()
        Jyy = F.conv2d(Jyy4, gaussian_kernel, padding=kernel_size // 2).squeeze()
        Jxy = F.conv2d(Jxy4, gaussian_kernel, padding=kernel_size // 2).squeeze()
        
        return Jxx, Jyy, Jxy
    
    def _get_gaussian_kernel(self, kernel_size, sigma, device, dtype):
        """Generate 2D Gaussian kernel"""
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= kernel_size // 2
        g = coords**2
        g = (-g / (2 * sigma**2)).exp()
        g /= g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x, mask=None, weight_map=None):
        """
        :param x: Input image [H, W]
        :param mask: Mask [H, W] (optional)
        :param weight_map: Weight map [H, W] (for weighted TV, optional)
        :return: TV loss scalar
        """
        L_PE = x.shape[0]
        L_RO = x.shape[1]
        
        # Compute first-order gradients
        dx = x[1:, :] - x[:L_PE - 1, :]
        dy = x[:, 1:] - x[:, :L_RO - 1]
        
        # Apply mask
        if mask is not None:
            dx = dx * mask[1:, :]
            dy = dy * mask[:, 1:]
        
        # Select different TV computation methods based on mode
        if self.mode == 'l1' or self.mode == 'anisotropic':
            # Anisotropic L1 TV
            tv_loss = (torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))) / ((L_PE - 1) * (L_RO - 1))
            
        elif self.mode == 'l2':
            # L2 TV
            tv_loss = (torch.sum(dx ** 2) + torch.sum(dy ** 2)) / ((L_PE - 1) * (L_RO - 1))
            
        elif self.mode == 'isotropic':
            # Isotropic TV: sqrt(dx^2 + dy^2)
            # Need to align dimensions
            min_h = min(dx.shape[0], dy.shape[0])
            min_w = min(dx.shape[1], dy.shape[1])
            dx_crop = dx[:min_h, :min_w]
            dy_crop = dy[:min_h, :min_w]
            gradient_mag = torch.sqrt(dx_crop ** 2 + dy_crop ** 2 + self.eps)
            tv_loss = torch.sum(gradient_mag) / (min_h * min_w)
            
        elif self.mode == 'charbonnier':
            # Charbonnier/Pseudo-Huber TV: sqrt(dx^2 + eps^2)
            charbonnier_dx = torch.sqrt(dx ** 2 + self.eps ** 2)
            charbonnier_dy = torch.sqrt(dy ** 2 + self.eps ** 2)
            tv_loss = (torch.sum(charbonnier_dx) + torch.sum(charbonnier_dy)) / ((L_PE - 1) * (L_RO - 1))
            
        elif self.mode == 'huber':
            # Huber TV (supports adaptive delta)
            abs_dx = torch.abs(dx)
            abs_dy = torch.abs(dy)
            
            delta_val = self._get_delta(abs_dx, abs_dy)
            delta_tensor = torch.tensor(delta_val, device=dx.device, dtype=dx.dtype)
            
            huber_dx = torch.where(
                abs_dx < delta_tensor,
                0.5 * dx ** 2 / delta_tensor,
                abs_dx - 0.5 * delta_tensor
            )
            huber_dy = torch.where(
                abs_dy < delta_tensor,
                0.5 * dy ** 2 / delta_tensor,
                abs_dy - 0.5 * delta_tensor
            )
            tv_loss = (torch.sum(huber_dx) + torch.sum(huber_dy)) / ((L_PE - 1) * (L_RO - 1))
            
        elif self.mode == 'lp':
            # Lp-TV (0 < p < 1, non-convex)
            min_h = min(dx.shape[0], dy.shape[0])
            min_w = min(dx.shape[1], dy.shape[1])
            dx_crop = dx[:min_h, :min_w]
            dy_crop = dy[:min_h, :min_w]
            gradient_mag = (dx_crop ** 2 + dy_crop ** 2 + self.eps) ** (self.p / 2.0)
            tv_loss = torch.sum(gradient_mag) / (min_h * min_w)
            
        elif self.mode == 'tgv':
            # Total Generalized Variation (simplified version: first-order + second-order TV)
            # First-order TV
            tv1 = torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))
            
            # Second-order differences
            dxx = dx[1:, :] - dx[:-1, :]
            dyy = dy[:, 1:] - dy[:, :-1]
            tv2 = torch.sum(torch.abs(dxx)) + torch.sum(torch.abs(dyy))
            
            # Weighted combination
            tv_loss = (self.alpha1 * tv1 / ((L_PE - 1) * (L_RO - 1)) + 
                      self.alpha2 * tv2 / max((L_PE - 2) * (L_RO - 2), 1))
            
        elif self.mode == 'weighted':
            # Weighted TV (using weight map or mask)
            if weight_map is not None:
                w = weight_map
            elif self.weight_map is not None:
                w = self.weight_map
            elif mask is not None:
                w = mask.float()
            else:
                w = torch.ones_like(x)
            
            # Apply weights to gradients
            w_dx = w[1:, :] * w[:-1, :]
            w_dy = w[:, 1:] * w[:, :-1]
            
            weighted_dx = w_dx * torch.abs(dx)
            weighted_dy = w_dy * torch.abs(dy)
            
            tv_loss = (torch.sum(weighted_dx) + torch.sum(weighted_dy)) / (torch.sum(w_dx) + torch.sum(w_dy) + 1e-8)
            
        elif self.mode == 'nonlocal':
            # Non-local TV (based on patch similarity, computationally intensive, simplified implementation)
            tv_loss = self._compute_nonlocal_tv(x, mask)
            
        elif self.mode == 'directional':
            # Directional/structure tensor TV
            Jxx, Jyy, Jxy = self._compute_structure_tensor(x)
            
            # Compute eigenvalues and eigenvectors
            trace = Jxx + Jyy
            det = Jxx * Jyy - Jxy * Jxy
            lambda1 = trace / 2 + torch.sqrt((trace / 2) ** 2 - det + self.eps)
            lambda2 = trace / 2 - torch.sqrt((trace / 2) ** 2 - det + self.eps)
            
            # Gradient weights are smaller along principal direction (preserves structure), larger in perpendicular direction (suppresses noise)
            # Simplified version here: eigenvalue-based weighting
            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + self.eps)
            
            # Crop dx, dy, coherence simultaneously to the same minimum size to avoid dimension mismatch
            min_h = min(dx.shape[0], dy.shape[0], coherence.shape[0])
            min_w = min(dx.shape[1], dy.shape[1], coherence.shape[1])

            dx_crop = torch.abs(dx[:min_h, :min_w])
            dy_crop = torch.abs(dy[:min_h, :min_w])
            coh_crop = coherence[:min_h, :min_w]

            # Apply directional weights: small weights along principal direction, large weights in perpendicular direction
            weighted_tv = (1 - coh_crop) * (dx_crop + dy_crop)
            tv_loss = torch.sum(weighted_tv) / (min_h * min_w)
            
        else:
            raise ValueError(f"Unknown TV mode: {self.mode}. Choose from ['l1', 'l2', 'huber', 'anisotropic', "
                           f"'isotropic', 'charbonnier', 'lp', 'tgv', 'weighted', 'nonlocal', 'directional']")
        
        return tv_loss
    
    def _compute_nonlocal_tv(self, x, mask=None):
        """Compute Non-local TV (simplified version)"""
        H, W = x.shape
        ps = self.patch_size
        sw = self.search_window
        
        # Use sparse sampling for simplified computation
        stride = max(ps // 2, 1)
        
        total_loss = 0.0
        count = 0
        
        # Extract patches
        for i in range(0, H - ps, stride):
            for j in range(0, W - ps, stride):
                if mask is not None and not mask[i:i+ps, j:j+ps].any():
                    continue
                    
                ref_patch = x[i:i+ps, j:j+ps]
                
                # Search for similar patches within search window
                i_min = max(0, i - sw // 2)
                i_max = min(H - ps, i + sw // 2)
                j_min = max(0, j - sw // 2)
                j_max = min(W - ps, j + sw // 2)
                
                for ii in range(i_min, i_max, stride):
                    for jj in range(j_min, j_max, stride):
                        if ii == i and jj == j:
                            continue
                        
                        comp_patch = x[ii:ii+ps, jj:jj+ps]
                        
                        # Compute similarity (using L2 distance)
                        similarity = torch.exp(-torch.sum((ref_patch - comp_patch) ** 2) / (ps * ps * self.eps))
                        
                        # Non-local difference
                        nl_diff = torch.sum(torch.abs(ref_patch - comp_patch)) * similarity
                        total_loss += nl_diff
                        count += 1
        
        return total_loss / (count + 1e-8)
    

class NablaBlochNet(nn.Module):
    
    def __init__(self, hidden_size=256, n_layers=4, output_dim=300):
        super(NablaBlochNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList()
        
        for pathway in range(3):
            # Input layer
            self.layers.append(nn.Linear(2, hidden_size))
            # Hidden layers
            for _ in range(n_layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            # Output layer
            self.layers.append(nn.Linear(hidden_size, output_dim))
        
        self.activation = nn.SiLU()
        
        self._pathway_size = n_layers + 1
    
    def _compute_pathway(self, x, pathway_idx):
        """Compute output for a specific network pathway."""
        start = pathway_idx * self._pathway_size
        end = start + self._pathway_size
        
        for i, layer_idx in enumerate(range(start, end)):
            x = self.layers[layer_idx](x)
            if i < self._pathway_size - 1:  # No activation on output layer
                x = self.activation(x)
        
        return x[:, :cfg.para_length]
    
    def forward(self, x):
        """
        Forward pass with physics-informed gradient computation.
        
        Args:
            x: Input tensor of shape (batch, 2) containing normalized T1/T2
            
        Returns:
            Signal prediction of shape (batch, n_timepoints)
        """
        return _NablaBackward.apply(x, self)       
    
    def load_weights(self, weight_path, device='cuda'):
        """Load pretrained weights from file."""
        state_dict = torch.load(weight_path, map_location=device)
        self.load_state_dict(state_dict)
        return self
    
    def freeze(self):
        """Freeze all parameters for inference."""
        for param in self.parameters():
            param.requires_grad = False
        return self
    
    def unfreeze(self):
        """Unfreeze parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        return self
