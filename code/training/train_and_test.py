#!/usr/bin/python
'''
Script for both training and evaluating the DLoc network
Automatically imports the parameters from params.py.
For further details onto which params file to load
read the README in `params_storage` folder.
'''

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import sys
import os
import torch
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
from utils import *
from modelADT import ModelADT
from Generators import *
from data_loader import load_data, LazyMultiHDF5Dataset
from joint_model import Enc_2Dec_Network
from joint_model import Enc_Dec_Network
from joint_model import Mamba_Network
from params import *
import trainer
# Seed is configurable for multi-seed runs (cross-session Table 1 experiments).
# Defaults to 0 to preserve original single-run behaviour.
_seed = int(os.environ.get('DLOC_SEED', 0))
torch.manual_seed(_seed)
np.random.seed(_seed)
print(f'[seed] DLOC_SEED={_seed}')

'''
Defining the paths from where to Load Data.
Assumes that the data is stored in a subfolder called data in the current data folder
'''

#####################################Final Simple Space Results################################################
if "data" in opt_exp and opt_exp.data == "rw_to_rw_atk":
    # Training and testing data loaded for the Final results For Env-1 (The smaller space) in the paper (Figure 10a)
    trainpath = ['./data/dataset_non_fov_train_July18.mat',
                './data/dataset_fov_train_July18.mat']
    testpath = ['./data/dataset_non_fov_test_July18.mat',
                './data/dataset_fov_test_July18.mat']
    print('Real World to Real World experiments started')

#####################################Final Complex Space Results################################################
elif "data" in opt_exp and opt_exp.data == "rw_to_rw":
    # Training and testing data loaded for the Final results For Env-2 (The larger space) in the paper (Figure 10b)
    trainpath = ['./data/dataset_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset_fov_test_jacobs_July28_2.mat',
                './data/dataset_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

#########################################Generalization across Scenarios###########################################

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env2":
    # Training and testing data loaded for the Final results For Env-2
    # for Generalization across scenarios (Table-1) train on 1/3/4 and test on 2
    trainpath = ['./data/dataset_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat',
                './data/dataset_train_jacobs_Aug16_3.mat',
                './data/dataset_train_jacobs_Aug16_4_ref.mat']
    testpath = ['./data/dataset_train_jacobs_Aug16_1.mat']
    print('Real World to Real World experiments started')


elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env3":
    # Training and testing data loaded for the Final results For Env-2
    # for Generalization across scenarios (Table-1) train on 1/2/4 and test on 3
    trainpath = ['./data/dataset_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat',
                './data/dataset_train_jacobs_Aug16_1.mat',
                './data/dataset_train_jacobs_Aug16_4_ref.mat']
    testpath = ['./data/dataset_train_jacobs_Aug16_3.mat']
    print('Real World to Real World experiments started')

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env4":
    # Training and testing data loaded for the Final results For Env-2
    # for Generalization across scenarios (Table-1) train on 1/2/3 and test on 4
    trainpath = ['./data/dataset_jacobs_July28.mat',
                './data/dataset_non_fov_train_jacobs_July28_2.mat',
                './data/dataset_fov_train_jacobs_July28_2.mat',
                './data/dataset_train_jacobs_Aug16_1.mat',
                './data/dataset_train_jacobs_Aug16_3.mat']
    testpath = ['./data/dataset_train_jacobs_Aug16_4_ref.mat']
    print('Real World to Real World experiments started')

######################################Generalization Across Bandwidth##########################################

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_40":
    # Training and testing data loaded for the Generalization results For Env-2 (The larger space) in the paper (Figure 13a) at 40MHz
    trainpath = ['./data/dataset40_jacobs_July28.mat',
                './data/dataset40_non_fov_train_jacobs_July28_2.mat',
                './data/dataset40_fov_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset40_fov_test_jacobs_July28_2.mat',
                './data/dataset40_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_20":
    # Training and testing data loaded for the Generalization results For Env-2 (The larger space) in the paper (Figure 13a) at 20MHz
    trainpath = ['./data/dataset20_jacobs_July28.mat',
                './data/dataset20_non_fov_train_jacobs_July28_2.mat',
                './data/dataset20_fov_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset20_fov_test_jacobs_July28_2.mat',
                './data/dataset20_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

######################################Generalization Across Space##########################################

elif "data" in opt_exp and opt_exp.data == "data_segment":
    # Training and testing data loaded for the Final results For Env-2 
    # for Disjoint Training and Testing(The larger space) in the paper (Figure 13b)
    trainpath = ['./data/dataset_train_jacobs_July28.mat',
                './data/dataset_train_jacobs_July28_2.mat']
    testpath = ['./data/dataset_test_jacobs_July28.mat',
                './data/dataset_test_jacobs_July28_2.mat']
    print('non-FOV to non-FOV experiments started')

######################################################################################################################
'''
Loading Training and Evaluation Data into RAM.
'''
print('Loading training files into RAM...')
train_wo, train_w, train_lbl = [], [], []
for fp in trainpath:
    wo, w, lbl = load_data(fp)
    if lbl.dim() == 3:
        lbl = lbl.unsqueeze(1)
    train_wo.append(wo)
    train_w.append(w)
    train_lbl.append(lbl)
train_wo = torch.cat(train_wo, dim=0)
train_w = torch.cat(train_w, dim=0)
train_lbl = torch.cat(train_lbl, dim=0)
train_data = torch.utils.data.TensorDataset(train_wo, train_w, train_lbl)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt_exp.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
print(f'# training samples = {len(train_data)}')
print(f'# training mini batch = {len(train_loader)}')

print('Loading testing files into RAM...')
test_wo, test_w, test_lbl = [], [], []
for fp in testpath:
    wo, w, lbl = load_data(fp)
    if lbl.dim() == 3:
        lbl = lbl.unsqueeze(1)
    test_wo.append(wo)
    test_w.append(w)
    test_lbl.append(lbl)
test_wo = torch.cat(test_wo, dim=0)
test_w = torch.cat(test_w, dim=0)
test_lbl = torch.cat(test_lbl, dim=0)
test_data = torch.utils.data.TensorDataset(test_wo, test_w, test_lbl)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=opt_exp.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
print(f'# testing samples = {len(test_data)}')
print(f'# testing mini batch = {len(test_loader)}')
print('Data Loaded (RAM)')

'''
Initiate the Network and build the graph
'''

model_type = getattr(opt_exp, 'model_type', 'dloc')

if model_type == 'mamba':
    # Mamba: single end-to-end model wrapped in one ModelADT
    from params import opt_mamba
    mamba_model = ModelADT()
    mamba_model.initialize(opt_mamba)
    mamba_model.setup(opt_mamba)

    print('Making the Mamba joint model')
    joint_model = Mamba_Network()
    joint_model.initialize(opt_exp, mamba_model, gpu_ids=opt_exp.gpu_ids)

else:
    # Original DLoc pipeline
    # init encoder
    enc_model = ModelADT()
    enc_model.initialize(opt_encoder)
    enc_model.setup(opt_encoder)

    # init decoder1
    dec_model = ModelADT()
    dec_model.initialize(opt_decoder)
    dec_model.setup(opt_decoder)

    if opt_exp.n_decoders == 2:
        # init decoder2
        offset_dec_model = ModelADT()
        offset_dec_model.initialize(opt_offset_decoder)
        offset_dec_model.setup(opt_offset_decoder)

        # join all models
        print('Making the joint_model')
        joint_model = Enc_2Dec_Network()
        joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)

    elif opt_exp.n_decoders == 1:
        # join all models
        print('Making the joint_model')
        joint_model = Enc_Dec_Network()
        joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

    else:
        print('Incorrect number of Decoders specified in the parameters')
        sys.exit(-1)

    if getattr(opt_exp, 'isFrozen', False):
        enc_model.load_networks(opt_encoder.starting_epoch_count)
        dec_model.load_networks(opt_decoder.starting_epoch_count)
        if opt_exp.n_decoders == 2:
            offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)

# train the model
'''
Trainig the network
'''
trainer.train(joint_model, train_loader, test_loader)

'''
Model Evaluation at the best epoch
'''

epoch = "best"  # int/"best"/"last"
eval_name = opt_exp.checkpoints_dir

# Check if "best" checkpoint exists; fall back to "latest" if not
if model_type == 'mamba':
    best_path = os.path.join(eval_name, 'mamba_dloc', f'best_net_mamba_dloc.pth')
else:
    best_path = os.path.join(eval_name, 'encoder', f'best_net_encoder.pth')
if not os.path.exists(best_path):
    print(f'WARNING: "best" checkpoint not found at {best_path}, falling back to "latest"')
    epoch = "latest"

if model_type == 'mamba':
    # Mamba: reload single model
    mamba_model.load_networks(epoch, load_dir=eval_name)
    joint_model.initialize(opt_exp, mamba_model, gpu_ids=opt_exp.gpu_ids)
else:
    # DLoc: reload encoder + decoder(s)
    enc_model.load_networks(epoch, load_dir=eval_name)
    dec_model.load_networks(epoch, load_dir=eval_name)
    if opt_exp.n_decoders == 2:
        offset_dec_model.load_networks(epoch, load_dir=eval_name)
        joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)
    elif opt_exp.n_decoders == 1:
        joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

# pass data through model
total_loss, median_error = trainer.test(joint_model,
    test_loader,
    save_output=True,
    save_dir=eval_name,
    save_name=f"decoder_test_result_epoch_{epoch}",
    log=False)
print(f"total_loss: {total_loss}, median_error: {median_error}")
