import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import os
import pickle
from module import Predictor, accuracy, auc
from matplotlib import pyplot as plt

def main():
    # reading in 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/sampling', help='determine the base dir of the dataset document')
    parser.add_argument("--sample_n", default=1000, type=int, help='starting image index of preprocessing')
    parser.add_argument("--evidence_n", default=25, type=int, help='how many top/bottom tiles to pick from')
    parser.add_argument("--repl_n", default=25, type=int, help='how many resampled replications')
    args = parser.parse_args()
    
    csv_file = 'data/useful_subset.csv'
    useful_subset = pd.read_csv(csv_file)
    gpu = True
    X = None
    Y = None
    X_cached_file = os.path.join(args.data_dir, 'X_%d.npy' % args.sample_n)
    Y_cached_file = os.path.join(args.data_dir, 'Y_%d.npy' % args.sample_n)
    if os.path.exists(X_cached_file) and os.path.exists(Y_cached_file):
        print('loading cached data')
        with open(X_cached_file,'rb') as file:
            X = np.load(file)
        with open(Y_cached_file,'rb') as file:
            Y = np.load(file)
    else:
        print('reading data from preprocessed file')
        useful_subset = pd.read_csv(csv_file)
        size = len(useful_subset)
        for i in range(1, size):
            for j in range(repl_n):
                X_this = np.loadtxt(args.data_dir+'/'+str(i)+'_'+str(j)+'_features.txt')
                Y_this = useful_subset.loc[i, 'outcome'] == 'good'
                if len(X_this)==args.sample_n:
                    if X == None:
                        X = X_this
                    else:
                        X = np.r_[X, X_this.reshape(1, args.sample_n, 2048)]
                    if Y == None:
                        Y = Y_this
                    else:
                        Y = np.r_[Y, Y_this]
                print("\r", "reading data input  %d/%d" % (i, size) , end='', flush=True)
        
        X = X.transpose((0, 2, 1))
        try:
            with open(X_cached_file,'wb') as file:
                np.save(file, X)
            with open(Y_cached_file,'wb') as file:
                np.save(file, Y)
        except Exception as e:
            print(e)

    if gpu:
        X, Y = torch.Tensor(X).cuda(), torch.Tensor(Y).cuda()
    # 5-folds cross validation
    perm = np.random.permutation(len(Y))
    id1, id2, id3, id4 = int(0.2 * len(Y)), int(0.4 * len(Y)), int(0.6 * len(Y)), int(0.8 * len(Y))
    perm_idx = np.array((perm[:id1], perm[id1:id2], perm[id2:id3], perm[id3:id4], perm[id4:]))

    n_epoch = 1000
    lr = 0.0005
    weight_decay = 0.05
    overfit = 0

    if not os.path.isdir('figure'):
        os.mkdir('figure')

    acc_folds = []
    auc_folds = []

    for i in range(5):
        train_idx_slice = [_ for _ in range(5)]
        train_idx_slice.remove(i)
        X_train = X[np.concatenate(perm_idx[train_idx_slice])]
        Y_train = Y[np.concatenate(perm_idx[train_idx_slice])]
        X_test = X[perm_idx[i]]
        Y_test = Y[perm_idx[i]]
        
        train_history = []
        test_history = []
        minimum_loss = None
        auc_fold = None
        acc_fold = None
        early_stop_count = 0
        
        model = Predictor(evidence_size=args.evidence_n, layers=[100, 50, 1])
        if gpu:
            model = model.cuda()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        print('starting fold %d' % i)
        
        for epoch in range(n_epoch):
            result = model(X_train)
            loss = nn.functional.binary_cross_entropy(result, Y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 20 == 0:
                result_test = model(X_test)
                loss_test = nn.functional.binary_cross_entropy(result_test, Y_test)
                train_history.append((epoch, loss, accuracy(result, Y_train), auc(result, Y_train)))
                test_history.append((epoch, loss_test, accuracy(result_test, Y_test), auc(result_test, Y_test)))
                print("%s epoch:%d loss:%.3f acc:%.3f auc:%.3f test_loss:%.3f test_acc:%.3f test_auc:%.3f" % 
                    (time.strftime('%m.%d %H:%M:%S', time.localtime(time.time())), epoch, loss, accuracy(result, Y_train), auc(result, Y_train), loss_test, accuracy(result_test, Y_test), auc(result_test, Y_test)))
                # early stop
                if minimum_loss is None or minimum_loss > loss_test:
                    minimum_loss = loss_test
                    auc_fold = auc(result_test, Y_test)
                    acc_fold = accuracy(result_test, Y_test)
                    early_stop_count = 0
                elif auc(result_test, Y_test) > auc_fold and auc(result_test, Y_test)>0.5 and accuracy(result_test, Y_test) >= acc_fold:
                    minimum_loss = loss_test
                    auc_fold = auc(result_test, Y_test)
                    acc_fold = accuracy(result_test, Y_test)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                if (early_stop_count > 3 and epoch>100 and auc_fold>0.55):
                    print('early stop at epoch %d' % epoch)
                    break
                if overfit == 0 and early_stop_count > 7:
                    weight_decay *= 2
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
                    overfit = 1

        train_history = np.array(train_history)
        test_history = np.array(test_history)
        acc_folds.append(acc_fold)
        auc_folds.append(auc_fold)
        plt.plot(train_history[:, 0], train_history[:, 1], label='train')
        plt.plot(test_history[:, 0], test_history[:, 1], label='test')
        plt.legend()
        plt.savefig('figure/sample_%d_fold%d.png' % (args.sample_n, i))
        plt.cla()
        print("acc:%.3f\tauc:%.3f"  % (acc_fold, auc_fold))

        del X_train, X_test, Y_train, Y_test
        torch.cuda.empty_cache()
    
    print('CV-auc:%.3f' % (sum(auc_folds) / 5))
    print('CV-acc:%.3f' % (sum(acc_folds) / 5))


if __name__ == '__main__':
    main()