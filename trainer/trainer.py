import os
import sys

from sklearn.metrics import f1_score

sys.path.append("..")
import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss, ClassBalancedLoss
import matplotlib.pyplot as plt
from utils import save_pic_cvs
torch.autograd.set_detect_anomaly(True)

def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, data_type, fuzzy_flag, instanceloss_flag, block, pot_number):
    # Start training
    logger.debug("Training started ....")
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    train_lossa, train_acca = [], []
    val_lossa, val_acca = [], []
    best_loss = float("inf")
    best_test_acc = float("-inf")
    best_chkpoint_s = None
    best_chkpoint_ns = None
    best_mf1 = float("-inf")
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        test_loss, test_acc, mf1, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, config)
        # 更新最佳测试准确率和模型参数
        if training_mode == "self_supervised" and train_loss < best_loss:
            best_chkpoint_s = {'model_state_dict': model.state_dict(),
                             'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
        if training_mode != "self_supervised" and test_acc > best_test_acc:
            best_chkpoint_ns = {'model_state_dict': model.state_dict(),
                             'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            best_test_acc = test_acc
            best_mf1 = mf1
        if training_mode != 'self_supervised':
            scheduler.step(test_loss)
        train_lossa.append(train_loss)
        # train_acca.append(train_acc)
        val_lossa.append(test_loss)
        # val_acca.append(valid_acc)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Test Loss     : {test_loss:.4f}\t | \tTest Accuracy     : {test_acc:2.4f}\n'
                     f'MF1    : {mf1:2.4f}')
    print(best_test_acc, best_mf1)
    logger.debug(f'Best model Test Acc     :{best_test_acc:0.4f}')
    if training_mode == "self_supervised":
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = best_chkpoint_s
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'best_model_s.pt'))
    if training_mode != "self_supervised":
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = best_chkpoint_ns
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'best_model_ns.pt'))
    myresult = 0.0
    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.

        logger.debug('\nEvaluate on the Test set:')
        model.load_state_dict(best_chkpoint_ns['model_state_dict'])
        temporal_contr_model.load_state_dict(best_chkpoint_ns['temporal_contr_model_state_dict'])

        # Re-evaluate on the test set
        test_loss, test_acc, mf1, _, _ = model_evaluate(model, temporal_contr_model, test_dl, criterion, training_mode, config)
        logger.debug(f'Best model Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')
        # # evaluate on the test set
        # logger.debug('\nEvaluate on the Test set:')
        # test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        myresult = test_acc
        # logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")
    if data_type == 'Aldata':
        pot_number_in = pot_number
    if fuzzy_flag == 'True':
        fuzzyi = 'fuzzy'
    else:
        fuzzyi = 'nofuzzy'
    if block == 'DilatedConv':  # DilatConv -> true ; mlp -> false
        blocki = 'DilatedConv'
    else:
        blocki = 'mlp'
    if instanceloss_flag == 'True':
        instanceloss = 'lossplus'
    else:
        instanceloss = 'origloss'
    file_path = 'D:\\python\\PycharmProjects\\TS-TCC-main\\result.xlsx'
    current_time = datetime.now().strftime('%m-%d,%H:%M')
    df = pd.read_excel(file_path, header=None)
    df.loc[df.shape[0] + 1, 0] = '_'.join([fuzzyi, blocki, instanceloss, data_type, current_time])
    df.loc[df.shape[0], 1] = myresult
    df.to_excel(file_path, index=False, header=False)

    # save_pic_cvs(data_type, training_mode, fuzzy, block, instanceloss, train_lossa, train_acca, val_lossa, val_acca)
    # save_picture of train&test loss
    if training_mode == "self_supervised" and fuzzy_flag =='True':
        plt.plot(train_lossa, label='Train')
        # plt.plot(val_loss, label='Valid')
        plt.title('self_supervised_train loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # 保存图像到文件
        plt.savefig('self_supervised_fuzzyture_train_loss.png')  # 这里你可以指定路径和文件名
        plt.close()  # 关闭当前图像，以防止后续的绘图覆盖当前图像
    if training_mode == "self_supervised" and fuzzy_flag !='True':
        plt.plot(train_lossa, label='Train')
        # plt.plot(val_loss, label='Valid')
        plt.title('self_supervised_train loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # 保存图像到文件
        plt.savefig('self_supervised_fuzzyfalse_train_loss.png')  # 这里你可以指定路径和文件名
        plt.close()  # 关闭当前图像，以防止后续的绘图覆盖当前图像
    if training_mode != "self_supervised" and fuzzy_flag != 'True':
        plt.plot(train_lossa, label='Test')
        # plt.plot(val_loss, label='Valid')
        plt.title('train_linear test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # 保存图像到文件
        plt.savefig('train_linear_fuzzyfalse_test_loss.png')  # 这里你可以指定路径和文件名
        plt.close()  # 关闭当前图像，以防止后续的绘图覆盖当前图像
    if training_mode != "self_supervised" and fuzzy_flag == 'True':
        plt.plot(train_lossa, label='Test')
        # plt.plot(val_loss, label='Valid')
        plt.title('train_linear test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # 保存图像到文件
        plt.savefig('train_linear_fuzzytrue_test_loss.png')  # 这里你可以指定路径和文件名
        plt.close()  # 关闭当前图像，以防止后续的绘图覆盖当前图像


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            # print("features1.shape:", features1.shape)   # torch.Size([64, 128, 182])
            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)
            # # ================= orig =========================
            # zis = temp_cont_lstm_feat1
            # zjs = temp_cont_lstm_feat2
            # ================== my ================
            zis = temp_cont_lstm_feat1.view(config.batch_size, -1, temp_cont_lstm_feat1.shape[-1])
            zjs = temp_cont_lstm_feat2.view(config.batch_size, -1, temp_cont_lstm_feat2.shape[-1])
        else:
            # print(data)
            # print(data.shape) torch.Size([batch_size, 1, feature_num])
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 0.5
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            # instance_loss = instance_contrastive_loss(features1, features2)
            loss = (temp_cont_loss1 + temp_cont_loss2) * (1-lambda1) + nt_xent_criterion(zis, zjs) * lambda1   # 1.0 0.7
            # loss = instance_loss * lambda1 + temporal_loss * (1-lambda1)
            
        else:  # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            # cb_loss = ClassBalancedLoss(label_counts.tolist(), config.num_classes)
            # loss = cb_loss.compute(labels, predictions)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
        total_loss.append(loss.item())
        loss.backward(retain_graph=True)
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, decive, training_mode, config):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()
    # criterion = LDAMLoss(class_counts)
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            # each class numbers
            # label_counts = torch.bincount(labels, minlength=config.num_classes)
            labels = labels.long()
            # # # 使用 torch.bincount 统计每个值的数量
            # counts = torch.bincount(labels, minlength=config.num_classes)
            # criterion = LDAMLoss(counts.numpy().tolist())
            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                # cb_loss = ClassBalancedLoss(label_counts.tolist(), config.num_classes)
                # loss = cb_loss.compute(labels, predictions)
                # print(loss)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        mf1 = 0
        return total_loss, total_acc, mf1, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
        # Calculate Macro F1 score
        macro_f1 = f1_score(trgs, outs, average='macro')
    return total_loss, total_acc, macro_f1, outs, trgs
