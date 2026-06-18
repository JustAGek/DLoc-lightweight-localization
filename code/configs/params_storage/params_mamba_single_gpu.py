# params_mamba_single_gpu.py
# ===========================
# Mamba-DLoc: replaces ResNet bottleneck with Mamba SSM blocks.
# Single decoder (location only, no consistency decoder).
#
# To use: copy this file to ../params.py
#   cp params_storage/params_mamba_single_gpu.py params.py
# Then run: python train_and_test.py
#
# Required data files in ./data/ (same as Fig 10b):
#   dataset_jacobs_July28.mat          (must be the FIXED 7,635-sample version)
#   dataset_fov_train_jacobs_July28_2.mat
#   dataset_non_fov_train_jacobs_July28_2.mat
#   dataset_fov_test_jacobs_July28_2.mat
#   dataset_non_fov_test_jacobs_July28_2.mat

from easydict import EasyDict as edict
import time
from os.path import join

opt_exp = edict()

# ---------- Global Experiment params ----------
opt_exp.isTrain = True
opt_exp.continue_train = False
opt_exp.starting_epoch_count = 0
opt_exp.n_epochs = 50
opt_exp.gpu_ids = ['0']
opt_exp.data = "rw_to_rw"           # Same Jacobs Hall dataset as Fig 10b
opt_exp.n_decoders = 1               # Single decoder (location only)
opt_exp.model_type = "mamba"         # Selects Mamba pipeline in train_and_test.py

opt_exp.batch_size = 32              # 64 OOMs on H100 with mamba-ssm
opt_exp.ds_step_trn = 1
opt_exp.ds_step_tst = 1
opt_exp.weight_decay = 1e-5
opt_exp.confidence = False

# ------ Experiment save paths ----------
opt_exp.save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
opt_exp.checkpoints_dir = join('./runs', opt_exp.save_name)
opt_exp.results_dir = opt_exp.checkpoints_dir
opt_exp.log_dir = opt_exp.checkpoints_dir
opt_exp.load_dir = opt_exp.checkpoints_dir

# ---------- Mamba model params (single model, acts as both encoder+decoder) ----------
opt_mamba = edict()
opt_mamba.parent_exp = opt_exp
opt_mamba.batch_size = opt_exp.batch_size
opt_mamba.ngf = 64
opt_mamba.base_model = 'mamba_dloc'
opt_mamba.net = 'G'
opt_mamba.no_dropout = False
opt_mamba.init_type = 'xavier'
opt_mamba.init_gain = 0.02
opt_mamba.norm = 'batch'
opt_mamba.beta1 = 0.5
opt_mamba.lr = 0.00001
opt_mamba.lr_policy = 'step'
opt_mamba.lr_decay_iters = 20
opt_mamba.lambda_L = 1
opt_mamba.lambda_cross = 0
opt_mamba.lambda_reg = 5e-4
opt_mamba.weight_decay = opt_exp.weight_decay

opt_mamba.input_nc = 4
opt_mamba.output_nc = 1
opt_mamba.save_latest_freq = 5000
opt_mamba.save_epoch_freq = 1
opt_mamba.n_epochs = opt_exp.n_epochs
opt_mamba.isTrain = True
opt_mamba.continue_train = False
opt_mamba.starting_epoch_count = opt_exp.starting_epoch_count
opt_mamba.name = 'mamba_dloc'
opt_mamba.loss_type = "L2_sumL1"
opt_mamba.niter = 20
opt_mamba.niter_decay = 100

opt_mamba.gpu_ids = opt_exp.gpu_ids
opt_mamba.num_threads = 4
opt_mamba.checkpoints_load_dir = opt_exp.load_dir
opt_mamba.checkpoints_save_dir = opt_exp.checkpoints_dir
opt_mamba.results_dir = opt_exp.results_dir
opt_mamba.log_dir = opt_exp.log_dir
opt_mamba.max_dataset_size = float("inf")
opt_mamba.verbose = False
opt_mamba.suffix = ''

# ---------- Aliases for train_and_test.py compatibility ----------
opt_encoder = opt_mamba
opt_decoder = opt_mamba
opt_offset_decoder = opt_mamba
