# params_fig10b_single_gpu.py
# ===========================
# Reproduces Fig 10b from the paper:
#   Complex environment - Jacobs Hall, July 28 + July 28_2, 4 APs
#   Target: median ~64 cm, 90th ~160 cm
#
# Adapted for single GPU (RTX 3060 6GB) with batch_size=8.
# To use: copy this file to ../params.py
#   cp params_storage/params_fig10b_single_gpu.py params.py
# Then run: python train_and_test.py
#
# Required data files in ./data/:
#   dataset_jacobs_July28.mat
#   dataset_fov_train_jacobs_July28_2.mat
#   dataset_non_fov_train_jacobs_July28_2.mat
#   dataset_fov_test_jacobs_July28_2.mat
#   dataset_non_fov_test_jacobs_July28_2.mat
# (Generate these with: python prepare_data.py)

from easydict import EasyDict as edict
import time
from os.path import join

opt_exp = edict()

# ---------- Global Experiment params ----------
opt_exp.isTrain = True
opt_exp.continue_train = False
opt_exp.starting_epoch_count = 0
opt_exp.n_epochs = 50
opt_exp.gpu_ids = ['0']           # Single GPU
opt_exp.data = "rw_to_rw"        # Jacobs Hall complex env
opt_exp.n_decoders = 2            # Encoder + location decoder + consistency decoder

opt_exp.batch_size = 64           # Larger batch for H100-class GPUs
opt_exp.ds_step_trn = 1
opt_exp.ds_step_tst = 1
opt_exp.weight_decay = 1e-5
opt_exp.confidence = False

# ------ Experiment save paths ----------
opt_exp.save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # No colons - Windows safe
opt_exp.checkpoints_dir = join('./runs', opt_exp.save_name)
opt_exp.results_dir = opt_exp.checkpoints_dir
opt_exp.log_dir = opt_exp.checkpoints_dir
opt_exp.load_dir = opt_exp.checkpoints_dir

# ---------- Encoder params ----------
opt_encoder = edict()
opt_encoder.parent_exp = opt_exp
opt_encoder.batch_size = opt_encoder.parent_exp.batch_size
opt_encoder.ngf = 64
opt_encoder.base_model = 'resnet_encoder'
opt_encoder.net = 'G'
opt_encoder.resnet_blocks = 6
opt_encoder.no_dropout = False
opt_encoder.init_type = 'xavier'
opt_encoder.init_gain = 0.02
opt_encoder.norm = 'instance'
opt_encoder.beta1 = 0.5
opt_encoder.lr = 0.00001
opt_encoder.lr_policy = 'step'
opt_encoder.lr_decay_iters = 50
opt_encoder.lambda_L = 1
opt_encoder.lambda_cross = 1e-5
opt_encoder.lambda_reg = 5e-4
opt_encoder.weight_decay = opt_encoder.parent_exp.weight_decay

opt_encoder.input_nc = 4         # 4 APs in Jacobs Hall
opt_encoder.output_nc = 1
opt_encoder.save_latest_freq = 5000
opt_encoder.save_epoch_freq = 1
opt_encoder.n_epochs = opt_encoder.parent_exp.n_epochs
opt_encoder.isTrain = True
opt_encoder.continue_train = False
opt_encoder.starting_epoch_count = opt_encoder.parent_exp.starting_epoch_count
opt_encoder.name = 'encoder'
opt_encoder.loss_type = "NoLoss"
opt_encoder.niter = 20
opt_encoder.niter_decay = 100

opt_encoder.gpu_ids = opt_encoder.parent_exp.gpu_ids
opt_encoder.num_threads = 4
opt_encoder.checkpoints_load_dir = opt_encoder.parent_exp.load_dir
opt_encoder.checkpoints_save_dir = opt_encoder.parent_exp.checkpoints_dir
opt_encoder.results_dir = opt_encoder.parent_exp.results_dir
opt_encoder.log_dir = opt_encoder.parent_exp.log_dir
opt_encoder.max_dataset_size = float("inf")
opt_encoder.verbose = False
opt_encoder.suffix = ''

# ---------- Location Decoder params ----------
opt_decoder = edict()
opt_decoder.parent_exp = opt_exp
opt_decoder.batch_size = opt_decoder.parent_exp.batch_size
opt_decoder.ngf = 64
opt_decoder.base_model = 'resnet_decoder'
opt_decoder.net = 'G'
opt_decoder.resnet_blocks = 9
opt_decoder.encoder_res_blocks = 6
opt_decoder.no_dropout = False
opt_decoder.init_type = 'xavier'
opt_decoder.init_gain = 0.02
opt_decoder.norm = 'instance'
opt_decoder.beta1 = 0.5
opt_decoder.lr = opt_encoder.lr
opt_decoder.lr_policy = 'step'
opt_decoder.lr_decay_iters = 20
opt_decoder.lambda_L = 1
opt_decoder.lambda_cross = 1e-5
opt_decoder.lambda_reg = 5e-4
opt_decoder.weight_decay = opt_decoder.parent_exp.weight_decay

opt_decoder.input_nc = 4         # 4 APs
opt_decoder.output_nc = 1        # 1-channel location heatmap
opt_decoder.save_latest_freq = 5000
opt_decoder.save_epoch_freq = 1
opt_decoder.n_epochs = opt_decoder.parent_exp.n_epochs
opt_decoder.isTrain = True
opt_decoder.continue_train = False
opt_decoder.starting_epoch_count = opt_decoder.parent_exp.starting_epoch_count
opt_decoder.name = 'decoder'
opt_decoder.loss_type = "L2_sumL1"
opt_decoder.niter = 20
opt_decoder.niter_decay = 100

opt_decoder.gpu_ids = opt_decoder.parent_exp.gpu_ids
opt_decoder.num_threads = 4
opt_decoder.checkpoints_load_dir = opt_decoder.parent_exp.load_dir
opt_decoder.checkpoints_save_dir = opt_decoder.parent_exp.checkpoints_dir
opt_decoder.results_dir = opt_decoder.parent_exp.results_dir
opt_decoder.log_dir = opt_decoder.parent_exp.log_dir
opt_decoder.verbose = False
opt_decoder.suffix = ''

# ---------- Consistency (Offset) Decoder params ----------
opt_offset_decoder = edict()
opt_offset_decoder.parent_exp = opt_exp
opt_offset_decoder.batch_size = opt_offset_decoder.parent_exp.batch_size
opt_offset_decoder.ngf = 64
opt_offset_decoder.base_model = 'resnet_decoder'
opt_offset_decoder.net = 'G'
opt_offset_decoder.resnet_blocks = 12
opt_offset_decoder.encoder_res_blocks = 6
opt_offset_decoder.no_dropout = False
opt_offset_decoder.init_type = 'xavier'
opt_offset_decoder.init_gain = 0.02
opt_offset_decoder.norm = 'instance'
opt_offset_decoder.beta1 = 0.5
opt_offset_decoder.lr = opt_encoder.lr
opt_offset_decoder.lr_policy = 'step'
opt_offset_decoder.lr_decay_iters = 50
opt_offset_decoder.lambda_L = 1
opt_offset_decoder.lambda_cross = 0
opt_offset_decoder.lambda_reg = 0
opt_offset_decoder.weight_decay = opt_offset_decoder.parent_exp.weight_decay

opt_offset_decoder.input_nc = 4  # 4 APs
opt_offset_decoder.output_nc = 4 # 4-channel consistency output
opt_offset_decoder.save_latest_freq = 5000
opt_offset_decoder.save_epoch_freq = 1
opt_offset_decoder.n_epochs = opt_offset_decoder.parent_exp.n_epochs
opt_offset_decoder.isTrain = True
opt_offset_decoder.continue_train = False
opt_offset_decoder.starting_epoch_count = opt_offset_decoder.parent_exp.starting_epoch_count
opt_offset_decoder.name = 'offset_decoder'
opt_offset_decoder.loss_type = "L2_offset_loss"
opt_offset_decoder.niter = 20
opt_offset_decoder.niter_decay = 100

opt_offset_decoder.gpu_ids = opt_offset_decoder.parent_exp.gpu_ids
opt_offset_decoder.num_threads = 4
opt_offset_decoder.checkpoints_load_dir = opt_offset_decoder.parent_exp.load_dir
opt_offset_decoder.checkpoints_save_dir = opt_offset_decoder.parent_exp.checkpoints_dir
opt_offset_decoder.results_dir = opt_offset_decoder.parent_exp.results_dir
opt_offset_decoder.log_dir = opt_offset_decoder.parent_exp.log_dir
opt_offset_decoder.verbose = False
opt_offset_decoder.suffix = ''
