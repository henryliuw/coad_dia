import numpy as np
import torch
import torch.nn as nn
import time
import os
from module import Predictor, CVDataLoader
from evaluation import accuracy, auc, c_index, recall, f1, precision
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # reading in 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/sampling', help='determine the base dir of the dataset document')
    parser.add_argument("--sample_n", default=1000, type=int, help='starting image index of preprocessing')
    parser.add_argument("--evidence_n", default=20, type=int, help='how many top/bottom tiles to pick from')
    parser.add_argument("--repl_n", default=3, type=int, help='how many resampled replications')
    parser.add_argument("--image_split", action='store_true', help='if use image_split')
    parser.add_argument("--batch_size", default=50, type=int, help="batch size")
    parser.add_argument("--stage_two", action='store_true', help='if only use stage two patients')
    parser.add_argument("--changhai", action='store_true', help='if use additional data')
    args = parser.parse_args()

    feature_size = 32
    #gpu = "cuda:0"
    gpu = None
    # 5-folds cross validation
    dataloader = CVDataLoader(args, gpu, feature_size)

    n_epoch = 800
    lr = 0.0005
    if args.stage_two:
        weight_decay = 0.008
    else:
        weight_decay = 0.005
    manytimes_n = 8

    if not os.path.isdir('figure'):
        os.mkdir('figure')
    if not os.path.isdir(os.path.join(args.data_dir, 'model')):
        os.mkdir(os.path.join(args.data_dir, 'model'))

    acc_folds = []
    auc_folds = []
    c_index_folds = []
    f1_folds = []
    f1_folds_pos = []
    total_round = 0
    model_count = 0

    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.8))

    for _ in range(manytimes_n): # averaging
        for i in range(5):
            train_history = []
            test_history = []
            minimum_loss = None
            auc_fold = None
            acc_fold = None
            early_stop_count = 0
            
            model = Predictor(evidence_size=args.evidence_n, layers=(100, 50, 1), feature_size=feature_size)
            # model.apply(weight_init)
            if gpu:
                model = model.to(gpu)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            dataloader.set_fold(i)
            X_test, Y_test, df_test = dataloader.get_test()
            # X_train, Y_train, df_train = dataloader.get_train()
            print('starting fold %d' % i)
            
            for epoch in range(n_epoch):
                #result = model(X_train)
                #loss = nn.functional.binary_cross_entropy(result, Y_train) + nn.functional.mse_loss(result, Y_train)
                # loss = nn.functional.mse_loss(result, Y_train)
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
                
                # batch input
                for X_train_batch, Y_train_batch, df_train_batch in dataloader:
                    # print(X_train_batch.shape)
                    result = model(X_train_batch)
                    loss = loss_function(result, Y_train_batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                X_train, Y_train, df_train = X_train_batch, Y_train_batch, df_train_batch

                if epoch % 20 == 0:
                    result_test = model(X_test)
                    loss_test = loss_function(result_test, Y_test)
                    #loss_test = nn.functional.mse_loss(result_test, Y_test)
                    acc_train, acc_test = accuracy(result, Y_train), accuracy(result_test, Y_test)
                    auc_train, auc_test = auc(result, Y_train),  auc(result_test, Y_test)
                    if args.changhai:
                        c_index_train, c_index_test = 0, 0
                    else:
                        c_index_train, c_index_test = c_index(result, df_train), c_index(result_test, df_test)
                    recall_train, recall_test = recall(result, Y_train),  recall(result_test, Y_test)
                    precision_train, precision_test = precision(result, Y_train),  precision(result_test, Y_test)
                    f1_train_pos, f1_test_pos = f1(result, Y_train),  f1(result_test, Y_test)
                    f1_train, f1_test = f1(result, Y_train, negative = True),  f1(result_test, Y_test, negative = True)
                    train_history.append((epoch, loss, acc_train, auc_train, c_index_train))
                    test_history.append((epoch, loss_test, acc_test, auc_test, c_index_test))
                    if epoch % 40 == 0:
                        print("%s epoch:%d loss:%.3f/%.3f acc:%.3f/%.3f auc:%.3f/%.3f c_index:%.3f/%.3f recall:%.3f/%.3f prec:%.3f/%.3f f1:%.3f/%.3f f1(neg):%.3f/%.3f" % 
                        (time.strftime('%m.%d %H:%M:%S', time.localtime(time.time())), epoch, loss,loss_test, acc_train,acc_test, auc_train,auc_test, c_index_train,c_index_test, recall_train, recall_test, precision_train, precision_test, f1_train_pos, f1_test_pos, f1_train, f1_test))
                    # early stop
                    if minimum_loss is None or minimum_loss * 0.995 > loss_test:
                    # if minimum_loss is None or minimum_loss > loss_test:
                        if f1_train == 0:
                            continue
                        minimum_loss = loss_test
                        auc_fold = auc_test
                        acc_fold = acc_test
                        c_index_fold = c_index_test
                        f1_fold_pos = f1_test_pos
                        f1_fold = f1_test
                        early_stop_count = 0
                    elif auc_test > auc_fold and auc_test>0.5 and acc_test >= acc_fold:
                        minimum_loss = loss_test
                        auc_fold = auc_test
                        acc_fold = acc_test
                        c_index_fold = c_index_test
                        f1_fold_pos = f1_test_pos
                        f1_fold = f1_test
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                    if early_stop_count > 2 and epoch>100:
                        if args.stage_two:
                            if auc_fold>0.55:
                                print('early stop at epoch %d' % epoch)
                                break
                        elif early_stop_count > 3:
                            print('early stop at epoch %d' % epoch)
                            break
                    if epoch > 500:
                        optimizer = torch.optim.RMSprop(model.parameters(), lr * 0.6, weight_decay=weight_decay * 1.2)

            train_history = np.array(train_history)
            test_history = np.array(test_history)
            acc_folds.append(acc_fold)
            auc_folds.append(auc_fold)
            f1_folds.append(f1_fold)
            f1_folds_pos.append(f1_fold_pos)
            c_index_folds.append(c_index_fold)
            plt.plot(train_history[:, 0], train_history[:, 1], label='train')
            plt.plot(test_history[:, 0], test_history[:, 1], label='test')
            plt.legend()
            plt.savefig('figure/sample_%d_fold%d.png' % (args.sample_n, i))
            plt.cla()
            if acc_fold > 0.7 and auc_fold > 0.6 and model_count < 10:
                model.save(args.data_dir + "/model/model_%d" % model_count)
                model_count += 1
            print("acc:%.3f\tauc:%.3f\tc_index:%.3f\tf1:%.3f"  % (acc_fold, auc_fold, c_index_fold, f1_fold))
            total_round += 1
            if gpu:
                del dataloader.X_train, dataloader.Y_train, dataloader.X_test, dataloader.Y_test
                del X_test, Y_test, X_train, Y_train, model, optimizer
                torch.cuda.empty_cache()
                
    print('CV-acc:%.3f CV-auc:%.3f CV-c-index:%.3f f1:%.3f f1(neg):%.3f' % (sum(acc_folds) / 5 / manytimes_n, sum(auc_folds) / 5 / manytimes_n, sum(c_index_folds) / 5 / manytimes_n, sum(f1_folds_pos) / 5 / manytimes_n, sum(f1_folds) / 5 / manytimes_n))


if __name__ == '__main__':
    main()