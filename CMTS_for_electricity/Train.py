# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import time
from matplotlib import pyplot
from CMTS_for_electricity.EvaluationMetrics import Evaluation_Utils
from CMTS_for_electricity import Const


def get_performance(preds, targets):
    # mae, mape, rmse = Evaluation_Utils.total(targets, preds)
    mae, rmse = Evaluation_Utils.total(targets.reshape(-1), preds.reshape(-1))
    return mae, rmse


def train_epoch(model, training_data, train_size, criterion, optimizer, device):
    total_loss = 0.0
    i = 0
    train_filtered_batch = training_data[0]
    train_target_batch = training_data[1]
    batchNum = len(train_filtered_batch)

    for b in range(batchNum):
        # print("batch:", b, 'in ', batchNum)
        start=time.time()
        # prepare data
        inputs = train_filtered_batch[b].to(device)
        # inputs = torch.tensor(temp[:, :, :1, :].clone().detach())
        target = train_target_batch[b].to(device)


        model.zero_grad()
        # forward
        # print('batch input size:', inputs.size())
        # print('batch input: ', inputs)
        preds = model(inputs,target)


        # backward
        loss = criterion(preds, target)
        loss.backward()

        # update parameters
        optimizer.step()

        total_loss += loss.item()*len(preds)
        i += 1
        # print("  - one batch data time: {:2.2f} min".format((time.time() - start)/60))
    return model, total_loss / train_size


def eval_epoch(model, dev_data, dev_size, criterion, device):
    total_loss = 0.0

    with torch.no_grad():
        dev_filtered_batch = dev_data[0]
        dev_target_batch = dev_data[1]
        batchNum = len(dev_filtered_batch)
        for b in range(batchNum):
            # print("batch:", b, 'in ', batchNum)
            start = time.time()
            inputs = dev_filtered_batch[b].to(device)
            # inputs = torch.tensor(temp[:, :, :1, :].clone().detach())
            target = dev_target_batch[b].to(device)
            preds = model.evaluate(inputs, Const.target_time_window)
            loss = criterion(preds, target)
            total_loss += loss.item()*len(preds)
            # print("  - one batch data time: {:2.2f} min".format((time.time() - start) / 60))
    return total_loss /dev_size


def train(model, dataAccess, criterion, optimizer, scheduler, option, device):
    log_train_file = None
    log_valid_file = None

    if option.log:
        log_train_file = option.log + ".train.log"
        log_valid_file = option.log + ".valid.log"

        print("[ INFO ] Training performance will be written to file\n {:s} and {:s}".format(
            log_train_file, log_valid_file))

    train_losses=[]
    valid_losses = []
    train_size = dataAccess.train_filtered_seq.size(0)
    dev_size = dataAccess.dev_filtered_seq.size(0)
    for each_epoch in range(option.epoch):
        model.train()
        print("[ Epoch {:d} ]".format(each_epoch))
        dataAccess.shuffle('train')
        [train_filtered_batch, train_target_batch] = dataAccess.divideIntoMiniBatch(
            ['train_filtered_seq', 'train_target_seq'], option.batch_size)

        dataAccess.shuffle('dev')
        [dev_filtered_batch, dev_target_batch] = dataAccess.divideIntoMiniBatch(
            ['dev_filtered_seq', 'dev_target_seq'], option.batch_size)

        start_time = time.time()
        model, train_loss = train_epoch(model, [train_filtered_batch, train_target_batch], train_size, criterion,
                                        optimizer, device)
        train_loss=np.sqrt(train_loss)
        print("  - (Training) loss (RMSE): {:2.8f},  elapse: {} seconds".format(train_loss, (time.time() - start_time)))

        start_time = time.time()
        eval_loss = eval_epoch(model, [dev_filtered_batch, dev_target_batch], dev_size, criterion, device)
        eval_loss=np.sqrt(eval_loss)
        print("  - (Validation) loss (RMSE): {:2.8f},  elapse: {} seconds".format(eval_loss, (time.time() - start_time)))

        scheduler.step()

        train_losses.append(train_loss)
        valid_losses += [eval_loss]

        model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "setting": option,
            "epoch": each_epoch
        }

        if option.save_model:
            if option.save_mode == "best":
                model_name = option.save_model + ".pkl"
                if eval_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print("  - [ INFO ] The checkpoint file has been updated.")
            elif option.save_mode == "all":
                model_name = option.save_model + "_loss_{:2.8f}.pkl".format(eval_loss)
                torch.save(checkpoint, model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as train_file, open(log_valid_file, "a") as valid_file:
                train_file.write("{}\t{:2.8f}\n".format(each_epoch, train_loss))
                valid_file.write("{}\t{:2.8f}\n".format(each_epoch, eval_loss))
        # if (each_epoch+1)%40==0:
        #     plotLossCurve(train_losses,valid_losses)

        # dynamic teacher forcing
        if Const.teacher_forcing_ratio>0.01:
            Const.teacher_forcing_ratio-=0.005
    # plotLossCurve(train_losses, valid_losses)


def test(model, dataAccess, criterion, device,case=0):
    # TODO: recover data and then compute the prediction error
    total_loss = 0.0

    with torch.no_grad():
        inputs = dataAccess.test_filtered_seq.to(device)
        # inputs = torch.tensor(temp[:, :, :1, :].clone().detach())
        targets = dataAccess.test_target_seq.to(device)
        preds = model.evaluate(inputs, Const.target_time_window)
        loss = criterion(preds, targets)
        total_loss += loss.item()
        # print('predict: ', preds.cpu().numpy())
        # print('real: ', targets.cpu().numpy())
        print('predict.size:',preds.size())
        print('real.size:',targets.size())
        predict = preds.cpu().numpy()
        real = targets.cpu().numpy()
        mae, rmse = get_performance(predict, real)
        # np.save('predict.npy', predict)
        # np.save('real.npy', real)
        # plotOneCase(predict,real,case)
    return mae, rmse, np.sqrt(total_loss)

def plotLossCurve(train_losses,valid_losses):
    pyplot.figure()
    pyplot.tick_params(labelsize=14)
    pyplot.plot(train_losses, color='r', label='train_losses')
    pyplot.plot(valid_losses, color='k', label='valid_losses')
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20, }
    pyplot.xlabel('Epoch', font)
    pyplot.ylabel('RMSE', font)
    pyplot.grid(linestyle='-.')
    # pyplot.title('RMSE:' + rmse + '  MAE:' + mae + '  R2_SCORE:' + r2)
    pyplot.legend()
    # pyplot.savefig('d:/Result.png')
    pyplot.show()

def plotOneCase(predict,real,sample):
    # plot real-predict line
    pyplot.figure()
    pyplot.tick_params(labelsize=14)
    pyplot.plot(predict[sample], marker='o', color='r', label='predict')
    pyplot.plot(real[sample], marker='o', color='k', label='real')
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20, }
    pyplot.xlabel('Time series', font)
    pyplot.ylabel('result', font)
    pyplot.grid(linestyle='-.')
    # pyplot.title('RMSE:' + rmse + '  MAE:' + mae + '  R2_SCORE:' + r2)
    pyplot.legend()
    # pyplot.savefig('d:/Result.png')
    pyplot.show()

if __name__=='__main__':
    target_time_window=18
    column=2
    log_train = '../resource/Result_Models/CMTS4elec/complete/{}h/CMTS@{}.{}.log'.format(
        target_time_window,column,'train')
    log_valid = '../resource/Result_Models/CMTS4elec/complete/{}h/CMTS@{}.{}.log'.format(
        target_time_window, column, 'valid')
    with open(log_train) as train_file, open(log_valid) as valid_file:
        li=train_file.read().strip().split('\n')
        train_losses=[]
        valid_losses=[]
        for item in li:
            ele=float(item.split('\t')[1])
            train_losses.append(ele)

        lv = valid_file.read().strip().split('\n')
        for item in lv:
            ele = float(item.split('\t')[1])
            valid_losses.append(ele)

    plotLossCurve(train_losses,valid_losses)
