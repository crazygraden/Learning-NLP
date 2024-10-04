import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy
import datetime

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)

    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)

    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    pd.set_option('display.max_columns', None)  # 显示所有列
    print("classification_report show:", df)

    df = df * 100

    output_file = 'result1.xlsx'
    df.to_excel(output_file, index=False)
    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger



def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))


# plot picture、save loss and acc in csv、save pictures
def save_pic_cvs(data_type, training_mode, fuzzy, block, instanceloss, train_lossa, train_acca, val_lossa, val_acca):
    if training_mode == "self_supervised":
        if fuzzy and block and instanceloss:
            filename = 'fuzzy_DilatConv_instanceloss_self_supervised.csv'
        elif fuzzy and block:
            filename = 'fuzzy_DilatConv_origloss_self_supervised.csv'
        elif fuzzy and instanceloss:
            filename = 'fuzzy_mlp_instanceloss_self_supervised.csv'
        elif block and instanceloss:
            filename = 'origDA_DilatConv_instanceloss_self_supervised.csv'
        elif instanceloss:
            filename = 'origDA_mlp_instanceloss_self_supervised.csv'
        elif block:
            filename = 'origDA_DilatConv_origloss_self_supervised.csv'
        elif fuzzy:
            filename = 'fuzzy_mlp_origloss_self_supervised.csv'
        else:
            filename = 'origDA_mlp_origloss_self_supervised.csv'
        # if exist, read csv
        if os.path.isfile(filename):
            ss = pd.read_csv(filename)
        else:
            ss = pd.DataFrame()
        # add new data in to DataFrame
        ss[data_type + datetime.datetime.now().strftime('%m-%d %H:%M:%S')] = train_lossa
        # save DataFrame to csv
        ss.to_csv(filename, index=False)
    else:
        if fuzzy and block and instanceloss:
            filename = 'fuzzy_DilatConv_instanceloss_train_linear.csv'
        elif fuzzy and block:
            filename = 'fuzzy_DilatConv_origloss_train_linear.csv'
        elif fuzzy and instanceloss:
            filename = 'fuzzy_mlp_instanceloss_train_linear.csv'
        elif block and instanceloss:
            filename = 'origDA_DilatConv_instanceloss_train_linear.csv'
        elif instanceloss:
            filename = 'origDA_mlp_instanceloss_train_linear.csv'
        elif block:
            filename = 'origDA_DilatConv_origloss_train_linear.csv'
        elif fuzzy:
            filename = 'fuzzy_mlp_origloss_train_linear.csv'
        else:
            filename = 'origDA_mlp_origloss_train_linear.csv'
        if os.path.isfile(filename):
            ft = pd.read_csv(filename)
        else:
            ft = pd.DataFrame()
        ft[data_type + "+t_l+" + datetime.datetime.now().strftime('%m-%d %H:%M:%S')] = train_lossa
        ft[data_type + "+t_a+" + datetime.datetime.now().strftime('%m-%d %H:%M:%S')] = train_acca
        ft[data_type + "+v_l+" + datetime.datetime.now().strftime('%m-%d %H:%M:%S')] = val_lossa
        ft[data_type + "+v_a+" + datetime.datetime.now().strftime('%m-%d %H:%M:%S')] = val_acca
        ft.to_csv(filename, index=False)


# plt.plot(train_lossa, label='Train')
#     # plt.plot(val_lossa, label='Valid')
#     plt.title('Train')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()