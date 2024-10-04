import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from trainer.my_trainer import My_Trainer
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

####################### Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='MotionSence', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD, Aldata')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cpu', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--model_select', default='mlp', type=str, help='mlp or DilatedConv')
parser.add_argument('--instance_loss', default='False', type=str)
parser.add_argument('--fuzzy', default='False', type=str)
parser.add_argument('--pot_period', default='None', type=str)
# parser.add_argument('--features_len', default='None', type=int)

args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description
fuzzy_flag = args.fuzzy
instanceloss_flag = args.instance_loss
block = args.model_select

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

if data_type == 'Aldata':
    config_file = 'Aldata5'
    exec(f'from config_files.{config_file}_Configs import Config as Configs')
else:
    exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs(args)
# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{data_type}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug(f'Fuzzy:   {configs.augmentation.fuzzy}')
logger.debug(f'instance_loss:   {args.instance_loss}')
logger.debug(f'4_convblock and final_out_channels({configs.final_out_channels})')
if args.model_select == 'DilatedConv':
    logger.debug('Attention + DilatedConv')
else:
    logger.debug('Attention +  MLP')
logger.debug("=" * 45)
# Load datasets
if args.pot_period == 'None':
    print("not my data")
    data_path = f"Learning-NLP/data/{data_type}"
else:
    print('my data')
    data_path = f"Learning-NLP/data/{data_type}/{args.pot_period}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")
# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "best_model_s.pt"), map_location=device)
    # print(chkpoint)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.


model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
if args.instance_loss == 'True':
    print("mytrainer")
    My_Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl,
               device, logger, configs, experiment_log_dir, training_mode, data_type, fuzzy_flag, instanceloss_flag, block)
else:
    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl,
               device, logger, configs, experiment_log_dir, training_mode, data_type, fuzzy_flag, instanceloss_flag, block, args.pot_period)

if training_mode != "self_supervised":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, configs)
    total_loss, total_acc, macro_f1, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")
