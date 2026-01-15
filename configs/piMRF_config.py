"""
Created on Thu Dec 23 2025
@author: Chaoguang Gong
This script defines configuration parameters for piMRF.
"""
import os
import time
import json

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
    print(f"[Config] Using GPU ID from environment variable: {gpu_id}")
else:
    gpu_id = "5"  # GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f"[Config] Using GPU ID from local_repository_profile: {gpu_id}")

import torch


# Data types 
dtype_float = torch.float
dtype_complex = torch.complex64
dtype_int = torch.int

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[Config] PyTorch device: {device}")

# Define loading function
def load_config(config_path: str = "config.json") -> dict:
    """Load and merge configuration
    - In `setting`, only need to provide `use_group` (group name), other runtime parameters are read from the corresponding `groups[group_name]`.
    - Do not write group fields back to the original `setting`; build `effective_setting` temporarily for merging.
    - Descriptive fields (e.g., UI text) are placed in `setting['_meta']`, do not participate in merge logic, but are returned as-is in the final configuration for external use.

    Returns: Merged configuration dictionary (including fields from common, fields expanded by module, direct fields from effective_setting, and optional `_meta` description dictionary).
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Top-level blocks
    setting = config.get("setting", {})
    groups = config.get("groups", {})
    common = config.get("common", {})

    # 1) Determine the group to use (prioritize use_group in setting)
    group_name = setting.get("use_group") or config.get("use_group")

    # 2) Construct effective_setting: retrieve imaging model related parameters from groups[group_name], and also get other direct runtime parameters from setting
    if group_name:
        if group_name not in groups:
            raise KeyError(f"use_group='{group_name}' not found in 'groups' of config.json. Available groups: {list(groups.keys())}")
        effective_setting = dict(groups[group_name] or {})
        # Get other direct runtime parameters from setting and append to effective_setting
        ALLOWED_OVERRIDES = {"code_label", "forward_Bloch_simulator", "display"}  # Fields allowed to override (or append)
        for k, v in setting.items():
            if k in ("use_group", "_meta"):
                continue
            if k not in ALLOWED_OVERRIDES:
                raise KeyError(f"Field '{k}' is not allowed to be overridden. Allowed fields: {sorted(ALLOWED_OVERRIDES)}")
            effective_setting[k] = v
    else:
        # If no group is specified, raise an error
        raise ValueError("use_group not specified in setting of config.json, and no default group provided. Please specify use_group in setting or define a default group in groups.")

    # 3) Assemble final merged: first add common, then expand modules according to effective_setting (such as model/sequence/csm etc.)
    merged = dict(common)

    for key, val in effective_setting.items():
        # Skip placeholder keys that are not modules
        if key in ("meta", "_meta"):
            continue

        # If this is a module key (e.g., 'model', 'sequence', 'ktraj', 'csm', etc.), expand from config[module][val]
        if key in config:
            module_dict = config[key]
            if val not in module_dict:
                raise KeyError(f"{key}='{val}' in setting not found in config['{key}']!")
            merged.update(module_dict[val])
        else:
            # Otherwise directly include this key-value as a runtime parameter in merged (e.g., some direct numerical values or paths)
            print(f"Warning: Setting field '{key}' not found in config.json, its value '{val}' has been directly included in the configuration.")
            merged[key] = val

    # 4) Preserve the intuitive fields of effective_setting (e.g., references like model=..., sequence=...) in merged for easy upper-level use
    #    Merging effective_setting here will not change the specific parameters after module expansion, it just puts the reference names into the returned dictionary
    merged.update(effective_setting)

    return merged

# Load configuration file
device_config = load_config("configs/config.json")

# Common general settings 
para_length = device_config.get('para_length', None) # Sequence length
seed_value = device_config.get('seed_value', None)  # Random seed
is_noisy = device_config.get('is_noisy', False)  # Whether to add noise
undksp_snr = device_config.get('undksp_snr', None)  # Undersampled k-space SNR
rank = device_config.get('rank', None)  # Dictionary rank

# Model data model type 
obj_type = device_config.get('obj_type', None)  # Object type
brain_size = device_config.get('img_size', None)  # Image size
data_path = device_config.get('data_path', None) # Data path
data_name = device_config.get('data_name', None) # Data name
slice_index = device_config.get('sliceIndex', None)  # Slice index
ref_map_path = device_config.get('refMap_path', None)  # Reference GT parameter path
ref_map_name = device_config.get('refMap_name', None)  # Reference GT parameter name
mask_path = device_config.get('mask_path', None)  # Mask path
mask_name = device_config.get('mask_name', None)  # Mask name
mask_for_NMSE_path = device_config.get('mask_for_NMSE_path', None)  # Mask path for NMSE calculation (skull removed)
mask_for_NMSE_name = device_config.get('mask_for_NMSE_name', None)  # Mask name for NMSE calculation (skull removed)
src_flag = device_config.get('src_flag', None)  # Data source: phantom, invivo_brain
scale_factor = device_config.get('scale_factor', None) # Data scaling factor
T1_max = device_config.get('T1_max', None) 
T2_max = device_config.get('T2_max', None) 
num_samples = None # Placeholder
dic_m = None  # Placeholder
LUT = None # Placeholder

# Sequence type 
sequence = device_config.get('sequence', None)  # Sequence name
FA_TR_path = device_config.get('FA_TR_path', None) # Sequence path
FA_type = device_config.get('FA_type', None)  # FA variable name
TR_type = device_config.get('TR_type', None)  # TR variable name
TE = device_config.get('TE', None)  # Echo Time

# CSM sensitivity map settings 
csm_map_path = device_config.get('csm_path', None)  # Sensitivity map path
csm_map_name = device_config.get('csm_name', None)  # Sensitivity map name
ncoil = device_config.get('num_coils', None)  # Number of receive coils

# Ktraj sampling template settings 
ktraj_und_path = device_config.get('ktraj_path', None)  # Undersampling template path
ktraj_und_name =  device_config.get('ktraj_name', None) # Undersampling template name
num_interleafs = device_config.get('num_interleaves', None)  # Number of interleaves for full sampling
num_spoke_perTR = device_config.get('num_spoke_perTR', None)  # Number of sampling spokes per TR
valid_start_samples = device_config.get('valid_start_samples', None)  # Valid sampling start point
valid_end_samples = device_config.get('valid_end_samples', None)  # Valid sampling end point

# Hyper_para_weight hyperparameter weights 
dc_weight = device_config.get('default_dc_weight', None) # Data consistency weight
csm_weight = device_config.get('default_csm_weight', None) # Sensitivity map consistency weight
tv_mode = device_config.get('tv_mode', None)  # TV regularization mode: recommend 'huber'
tv_total_weight = device_config.get('default_tv_total_weight', None) # TV regularization total weight
tv_weight = torch.tensor(device_config.get('default_tv_weight', None), dtype=dtype_float, device=device) * tv_total_weight # TV regularization weights for T1, T2, and PD respectively

# NUFFT parameter settings
grid_factor = device_config.get('grid_factor', 2)  # Grid factor, used for gridding of nufft sampling template
numpoints = device_config.get('numpoints', 6)  # Number of interpolation points, used for nufft sampling template interpolation
norm = device_config.get('norm', 'ortho')  # Normalization method
torch_batch_size_L = para_length // 2  # Number of time frames per batch when doing nufft on ksp_und
torch_batch_size_k = rank // 0.9  # Number of time frames per batch when doing nufft on ksp_full
bloch_sim_batch_size = brain_size[0] * brain_size[1] // 0.9 # Batch size for image domain Bloch simulation

# forward_Bloch_simulator Forward Bloch simulator 
forward_bloch_simulator_type = device_config.get('simulator_type', None)  # Forward Bloch simulator type
nabla_BlochNet_path = device_config.get('nabla_BlochNet_path', None) # Pretrained nabla-BlochNet model weight file path

# hyper_para_backbone Backbone network hyperparameters 
unsuper_Iter_num = device_config.get('unsuper_Iter_num', None)  # Unsupervised training iteration number
img_save_pre = device_config.get('img_save_pre', None)  # Output prediction image interval
backup_interval = device_config.get('backup_interval', None)  # Model output backup interval
unsuper_LrRate = device_config.get('unsuper_LrRate', None)  # Unsupervised training learning rate
lr_scheduler_ParaMaps_step_size = device_config.get('lr_scheduler_ParaMaps_step_size', None)  # Learning rate adjustment step size for T1, T2, PD
lr_scheduler_ParaMaps_gamma = device_config.get('lr_scheduler_ParaMaps_gamma', None)  # Learning rate adjustment factor for T1, T2, PD
lr_scheduler_CSMaps_step_size = device_config.get('lr_scheduler_CSMaps_step_size', None)  # Learning rate adjustment step size for CSM
lr_scheduler_CSMaps_gamma = device_config.get('lr_scheduler_CSMaps_gamma', None)  # Learning rate adjustment factor for CSM

# Model output parameter mapping range 
T1_min = device_config.get('T1_min', None) 
T2_min = device_config.get('T2_min', None) 
PD_min = device_config.get('PD_min', None)
PD_max = device_config.get('PD_max', None)

# Value range of signal-derivative dictionary
dic_m_min = device_config.get('dic_m_min', None)
dic_m_max = device_config.get('dic_m_max', None)
dic_dmdt1_min = device_config.get('dic_dmdt1_min', None)
dic_dmdt1_max = device_config.get('dic_dmdt1_max', None)
dic_dmdt2_min = device_config.get('dic_dmdt2_min', None)
dic_dmdt2_max = device_config.get('dic_dmdt2_max', None)

# Model backbone network parameters 
backbone_in_features = device_config.get('backbone_in_features', None) # Backbone network input feature dimension
backbone_hidden_features = device_config.get('backbone_hidden_features', None) # Backbone network hidden layer feature dimension
backbone_hidden_layers = device_config.get('backbone_hidden_layers', None) # Backbone network hidden layer count
backbone_out_features = device_config.get('backbone_out_features', None) # Backbone network output feature dimension

# display Visualization parameters 
colormap = device_config.get('colormap', None) # Visualization colormap

# Output paths 
code_label = device_config.get('code_label', None) + src_flag # Test code name
unique_flag = time.strftime("%m%d%H%M%S", time.localtime(time.time())) # Generate unique flag variable
print('The unique flag is: ', '(' + unique_flag + ')')

unique_suffix_signature = (
    f"{unique_flag}_{src_flag}_S{slice_index}"
    f"_L{para_length}_c{ncoil}_Ite{unsuper_Iter_num}_{undksp_snr}dB"
    f"_TVw{int(tv_total_weight)}_DCw{int(dc_weight)}_CSMw{int(csm_weight)}"
) # Output suffix signature

# Define paths and create required directories
base_results_dir = './results'

if not os.path.exists(base_results_dir):
    os.makedirs(base_results_dir, exist_ok=True)

Save_NET_root = os.path.join(base_results_dir, 'Model_save')
save_img_root = os.path.join(base_results_dir, 'imgs')
result_archive_root = os.path.join(base_results_dir, 'outputs', code_label)  # Archive path for final reconstruction results
temp_img_save_boot = os.path.join(save_img_root, code_label, unique_suffix_signature)  # Save path for intermediate display images during reconstruction
log_file_path = os.path.join(temp_img_save_boot, "log.txt")  # Log file path

# Create directories
for path in [Save_NET_root, save_img_root, result_archive_root, temp_img_save_boot, os.path.dirname(log_file_path)]:
    os.makedirs(path, exist_ok=True)

# Write specified parameters to log.txt file
log_data = [
    "######### Configuration Log #########",
    f"---------Code_label: {code_label}",
    f"---------Model: {device_config.get('model', None)}",
    f"---------Hyper_para_loss_weight: {device_config.get('hyper_para_loss_weight', None)}",
    f"---------Hyper_para_backbone: {device_config.get('hyper_para_backbone', None)}",
    f"---------Forward_Bloch_simulator: {device_config.get('forward_Bloch_simulator', None)}",
    f"---------Sequence: {device_config.get('sequence', None)}",
    f"---------Csm: {device_config.get('csm', None)}",
    f"---------Ktraj: {device_config.get('ktraj', None)}",
    f"---------Display: {device_config.get('display', None)}",
    f"---------Tv mode: {tv_mode}",
    "","",
    "######### Dashboard Overview #########",
    f"---------Source Flag: {src_flag}",
    f"---------Slice Index: {slice_index}",
    f"---------T1 Range: {T1_min} - {T1_max}",
    f"---------T2 Range: {T2_min} - {T2_max}",
    f"---------Data Consistency Weight (DC): {dc_weight}",
    f"---------TV Regularization total Weight: {tv_total_weight}",
    f"---------TV Regularization Weights: {tv_weight.tolist()}",
    f"---------CSM Weight: {csm_weight}",
    f"---------Scale Factor: {scale_factor}",
    f"---------L Length: {para_length}",
    "","",
    "######### Task Settings #########",
    f"---------Unique_flag: {unique_flag}",
    f"---------Unique_suffix_signature: {unique_suffix_signature}",
    f"---------Save_NET_root: {Save_NET_root}",
    f"---------save_img_root: {save_img_root}",
    f"---------result_archive_root: {result_archive_root}",
    f"---------temp_img_save_boot: {temp_img_save_boot}",
    f"---------log_file_path: {log_file_path}",
    "","",
    "######### Common Settings #########",
    f"---------Para_length: {para_length}",
    f"---------Seed_value: {seed_value}",
    f"---------Is_noisy: {is_noisy}",
    f"---------Undksp_snr: {undksp_snr}",
    f"---------Rank: {rank}",
    "","",
    "######### Data Model Settings #########",
    f"---------Obj_type: {obj_type}",
    f"---------Img_size: {brain_size}",
    f"---------Data_path: {data_path}",
    f"---------Data_name: {data_name}",
    f"---------SliceIndex: {slice_index}",
    f"---------RefMap_path: {ref_map_path}",
    f"---------RefMap_name: {ref_map_name}",
    f"---------Mask_path: {mask_path}",
    f"---------Mask_name: {mask_name}",
    f"---------Mask_for_NMSE_path: {mask_for_NMSE_path}",
    f"---------Mask_for_NMSE_name: {mask_for_NMSE_name}",
    f"---------Src_flag: {src_flag}",
    f"---------Scale_factor: {scale_factor}",
    f"---------T1_max: {T1_max}",
    f"---------T2_max: {T2_max}",
    "", "",
    "######### Sequence Settings #########",
    f"---------Sequence: {sequence}",
    f"---------FA_TR_path: {FA_TR_path}",
    f"---------FA_type: {FA_type}",
    f"---------TR_type: {TR_type}",
    f"---------TE: {TE}",
    "","",
    "######### CSM Settings #########",
    f"---------Csm_path: {csm_map_path}",
    f"---------Csm_name: {csm_map_name}",
    f"---------Num_coils: {ncoil}",
    "","",
    "######### Ktraj Settings #########",
    f"---------Ktraj_path: {ktraj_und_path}",
    f"---------Ktraj_name: {ktraj_und_name}",
    f"---------Num_interleaves: {num_interleafs}",
    f"---------Num_spoke_perTR: {num_spoke_perTR}",
    f"---------Valid_start_samples: {valid_start_samples}",
    f"---------Valid_end_samples: {valid_end_samples}",
    "","",
    "######### Hyper_para_weight Settings #########",
    f"---------Dc_data_weight: {dc_weight}",
    f"---------DC_csm_weight: {csm_weight}",
    f"---------Tv_mode: {tv_mode}",
    f"---------Tv_weight: {tv_weight.cpu().numpy().tolist()}",
    "","",
    "######### Hyper_para_backbone Settings #########",
    f"---------Unsuper_Iter_num: {unsuper_Iter_num}",
    f"---------Img_save_pre: {img_save_pre}",
    f"---------Backup_interval: {backup_interval}",
    f"---------Unsuper_LrRate: {unsuper_LrRate}",
    f"---------Lr_scheduler_ParaMaps_step_size: {lr_scheduler_ParaMaps_step_size}",
    f"---------Lr_scheduler_ParaMaps_gamma: {lr_scheduler_ParaMaps_gamma}",
    f"---------lr_scheduler_CSMaps_step_size: {lr_scheduler_CSMaps_step_size}",
    f"---------lr_scheduler_CSMaps_gamma: {lr_scheduler_CSMaps_gamma}",
    f"---------T1 Range: {T1_min} - {T1_max}",
    f"---------T2 Range: {T2_min} - {T2_max}",
    f"---------T1_min: {T1_min}",
    f"---------T1_max: {T1_max}",
    f"---------T2_min: {T2_min}",
    f"---------T2_max: {T2_max}",
    f"---------PD_min: {PD_min}", 
    f"---------PD_max: {PD_max}",
    f"---------Dic_m_min: {dic_m_min}",
    f"---------Dic_m_max: {dic_m_max}",
    f"---------Dic_dmdt1_min: {dic_dmdt1_min}",
    f"---------Dic_dmdt1_max: {dic_dmdt1_max}",
    f"---------Dic_dmdt2_min: {dic_dmdt2_min}",
    f"---------Dic_dmdt2_max: {dic_dmdt2_max}",
    "","",
    "######### Transform_method Settings #########",
    f"---------Grid_factor: {grid_factor}",
    f"---------Numpoints: {numpoints}",
    f"---------Norm: {norm}",
    f"---------Torch_batch_size_L: {torch_batch_size_L}",
    f"---------Torch_batch_size_k: {torch_batch_size_k}",
    f"---------Bloch_sim_batch_size: {bloch_sim_batch_size}",
    "","",
    "######### Forward_Bloch_simulator Settings #########",
    f"---------Simulator_type: {forward_bloch_simulator_type}",
    f"---------NablaBlochNet_path: {nabla_BlochNet_path}",
    "######################################\n"
    "","",    
]    

with open(log_file_path, "w") as log_file:
    log_file.write('\n'.join(log_data))

print(f"Configuration parameters have been written to {log_file_path}")
