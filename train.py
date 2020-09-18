import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import os
import pickle
from module import Predictor, accuracy, auc, c_index, CVDataLoader, weight_init
from matplotlib import pyplot as plt

def main():
    # reading in 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/sampling', help='determine the base dir of the dataset document')
    parser.add_argument("--sample_n", default=1000, type=int, help='starting image index of preprocessing')
    parser.add_argument("--evidence_n", default=20, type=int, help='how many top/bottom tiles to pick from')
    parser.add_argument("--repl_n", default=3, type=int, help='how many resampled replications')
    parser.add_argument("--image_split", action='store_true', help='if use image_split')
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    args = parser.parse_args()

    image_split = args.image_split
    gpu = True
    X = None
    Y = None
    X_cached_file = os.path.join(args.data_dir, 'X_%d.npy' % args.sample_n)
    Y_cached_file = os.path.join(args.data_dir, 'Y_%d.npy' % args.sample_n)
    df_cached_file = os.path.join(args.data_dir, 'df.csv')
    if os.path.exists(X_cached_file) and os.path.exists(Y_cached_file):
        print('loading cached data')
        with open(X_cached_file,'rb') as file:
            X = np.load(file)
        with open(Y_cached_file,'rb') as file:
            Y = np.load(file)
        df_new = pd.read_csv(df_cached_file)
    else:
        df_new = None
        print('reading data from preprocessed file')
        csv_file = 'data/useful_subset.csv'
        useful_subset = pd.read_csv(csv_file)
        useful_subset['OS.time'].fillna(useful_subset['OS.time'].mean(), inplace=True)
        size = len(useful_subset)
        for i in range(1, size):
            for j in range(args.repl_n):
                if args.repl_n == 0:
                    X_this = np.loadtxt(args.data_dir+'/'+str(i)+'_features.txt')
                else:
                    X_this_file = args.data_dir+'/'+str(i)+'_'+str(j)+'_features.txt'
                    if os.path.exists(X_this_file):
                        X_this = np.loadtxt(X_this_file)
                    else:
                        continue
                Y_this = useful_subset.loc[i, 'outcome'] == 'good'
                if len(X_this)==args.sample_n:
                    if X is None:
                        X = X_this.reshape(1, args.sample_n, 2048)
                    else:
                        X = np.r_[X, X_this.reshape(1, args.sample_n, 2048)]
                    if Y is None:
                        Y = Y_this
                    else:
                        Y = np.r_[Y, Y_this]
                    image_file = args.data_dir+'/'+str(i)+'_'+str(j)+'_name.pkl'
                    if df_new is None:
                        df_new = pd.DataFrame({"y": Y_this, "time": useful_subset.loc[i, "OS.time"], 'sample_id':i, 'image_file':image_file}, index=[0])
                    else:
                        df_new = df_new.append({"y": Y_this, "time": useful_subset.loc[i, "OS.time"], 'sample_id':i, 'image_file':image_file}, ignore_index=True)
                print("\r", "reading data input  %d/%d" % (i, size) , end='', flush=True)
        
        X = X.transpose((0, 2, 1))
        try:
            with open(X_cached_file,'wb') as file:
                np.save(file, X)
            with open(Y_cached_file,'wb') as file:
                np.save(file, Y)
            df_new.to_csv(df_cached_file)
        except Exception as e:
            print(e)

    # 5-folds cross validation
    dataloader = CVDataLoader(X, Y, df_new, args.repl_n, gpu, image_split)

    n_epoch = 800
    lr = 0.0005
    weight_decay = 0.03
    overfit = 0
    manytimes_n = 2

    if not os.path.isdir('figure'):
        os.mkdir('figure')

    acc_folds = []
    auc_folds = []
    c_index_folds = []
    total_round = 0

    for _ in range(manytimes_n):
        for i in range(5):
            train_history = []
            test_history = []
            minimum_loss = None
            auc_fold = None
            acc_fold = None
            early_stop_count = 0
            
            model = Predictor(evidence_size=args.evidence_n, layers=[100, 50, 1])
            # model.apply(weight_init)
            if gpu:
                model = model.cuda()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            
            dataloader.set_fold(i)
            X_test, Y_test, df_test = dataloader.get_test()
            X_train, Y_train, df_train = dataloader.get_train()
            print('starting fold %d' % i)
            
            for epoch in range(n_epoch):
                result = model(X_train)
                loss = nn.functional.binary_cross_entropy(result, Y_train) + nn.functional.mse_loss(result, Y_train)
                # loss = nn.functional.mse_loss(result, Y_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # batch input
                #for X_train_batch, Y_train_batch, df_train_batch in dataloader:
                #    result = model(X_train_batch)
                #    loss = nn.functional.binary_cross_entropy(result, Y_train_batch)
                #    loss.backward()
                #    optimizer.step()
                #    optimizer.zero_grad()

                if epoch % 25 == 0:
                    result_test = model(X_test)
                    loss_test = nn.functional.binary_cross_entropy(result_test, Y_test) + nn.functional.mse_loss(result_test, Y_test)
                    #loss_test = nn.functional.mse_loss(result_test, Y_test)
                    acc_train, acc_test = accuracy(result, Y_train), accuracy(result_test, Y_test)
                    auc_train, auc_test = auc(result, Y_train),  auc(result_test, Y_test)
                    c_index_train, c_index_test = c_index(result, df_train), c_index(result_test, df_test)
                    train_history.append((epoch, loss, acc_train, auc_train, c_index_train))
                    test_history.append((epoch, loss_test, acc_test, auc_test, c_index_test))
                    print("%s epoch:%d loss:%.3f acc:%.3f auc:%.3f c_index:%.3f test_loss:%.3f test_acc:%.3f test_auc:%.3f test_c_index:%.3f" % 
                        (time.strftime('%m.%d %H:%M:%S', time.localtime(time.time())), epoch, loss, acc_train, auc_train, c_index_train, loss_test, acc_test, auc_test, c_index_test))
                    # early stop
                    if minimum_loss is None or minimum_loss * 0.995 > loss_test:
                        minimum_loss = loss_test
                        auc_fold = auc_test
                        acc_fold = acc_test
                        c_index_fold = c_index_test
                        early_stop_count = 0
                    elif auc_test > auc_fold and auc_test>0.5 and acc_test >= acc_fold:
                        minimum_loss = loss_test
                        auc_fold = auc_test
                        acc_fold = acc_test
                        c_index_fold = c_index_test
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                    if (early_stop_count > 2 and epoch>100 and auc_fold>0.55):
                        print('early stop at epoch %d' % epoch)
                        break
                    if epoch > 500:
                        optimizer = torch.optim.RMSprop(model.parameters(), lr * 0.8, weight_decay=weight_decay)
                        overfit = 1

            train_history = np.array(train_history)
            test_history = np.array(test_history)
            acc_folds.append(acc_fold)
            auc_folds.append(auc_fold)
            c_index_folds.append(c_index_fold)
            plt.plot(train_history[:, 0], train_history[:, 1], label='train')
            plt.plot(test_history[:, 0], test_history[:, 1], label='test')
            plt.legend()
            plt.savefig('figure/sample_%d_fold%d.png' % (args.sample_n, i))
            plt.cla()
            model.save(args.save_dir)
            print("acc:%.3f\tauc:%.3f\tc_index:%.3f"  % (acc_fold, auc_fold, c_index_fold))
            total_round += 1
            if gpu:
                del dataloader.X_train, dataloader.Y_train, dataloader.X_test, dataloader.Y_test
                del X_test, Y_test, X_train, Y_train, model, optimizer
                torch.cuda.empty_cache()
                
    print('CV-acc:%.3f CV-auc:%.3f CV-c-index:%.3f' % (sum(acc_folds) / 5 / manytimes_n, sum(auc_folds) / 5 / manytimes_n, sum(c_index_folds) / 5 / manytimes_n))


if __name__ == '__main__':
    main()