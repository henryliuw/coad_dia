from module import Predictor, CVDataLoader, GNN, CrossValidationSplitter
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.data import DataLoader
import random
from evaluation import accuracy, auc, c_index, recall, f1, precision

def to_PyG_graph(data, model, image_file, y, threshold=80, evidence_n=20):
    
    score = model.tile_scoring(data.reshape(1,32,2000)).view(2000).detach()
    with open(image_file, 'rb') as file:
        location = pickle.load(file)
    bot_idx = np.argsort(score)[-evidence_n:]
    top_idx = np.argsort(score)[:evidence_n]
    all_idx = np.r_[bot_idx, top_idx]
    location_mat = np.array(location)[all_idx]

    dist_1 = location_mat[:, 1].reshape(-1,1) - location_mat[:,1].reshape(1,-1)
    dist_0 = location_mat[:, 0].reshape(-1,1) - location_mat[:,0].reshape(1,-1)
    dist = ((dist_0 ** 2 + dist_1 ** 2) ** 0.5)
    A = np.zeros_like(dist, dtype=np.int)
    A [(dist < threshold)] = 1
    A[range(evidence_n * 2),range(evidence_n * 2)] = 0
    # print("Average node degree:%.2f" % (A.sum() / 2.0 /evidence_n))

    edge_index = []
    for i in range(40):
        for j in range(40):
            if A[i, j]:
                edge_index.append((i,j))
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    x = data[:, all_idx].t().contiguous()
    # x = score[all_idx].reshape(2 * evidence_n, 1)
    graph_data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float)) 
    
    return graph_data

def construct_graph_dataset(args):
    dataloader = CVDataLoader(args, gpu=None, feature_size=32)
    dataloader.df.reset_index(inplace=True)
    model = Predictor(evidence_size=20, layers=(100, 50, 1), feature_size=32)
    model.load(args.data_dir+'/model/model_0')
    dataset = []
    for i in dataloader.df.index:
        data = dataloader.X[i]
        image_file =  dataloader.df.loc[i, 'image_file']
        dataset.append(to_PyG_graph(data, model, image_file, dataloader.df.loc[i, 'y2']))
    
    return dataset, dataloader.df

def concat_result(dataloader, model):
    out = []
    target = []
    for data in dataloader:
        out.append(model(data.x, data.edge_index, data.batch).view(-1))
        target.append(data.y)
    return torch.cat(out), torch.cat(target)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/sampling', help='determine the base dir of the dataset document')
    parser.add_argument("--sample_n", default=1000, type=int, help='starting image index of preprocessing')
    parser.add_argument("--evidence_n", default=20, type=int, help='how many top/bottom tiles to pick from')
    parser.add_argument("--repl_n", default=3, type=int, help='how many resampled replications')
    parser.add_argument("--image_split", action='store_true', help='if use image_split')
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--stage_two", action='store_true', help='if only use stage two patients')
    parser.add_argument("--changhai", action='store_true', help='if use additional data')
    args = parser.parse_args()

    n_epoch = 80
    acc_folds = []
    auc_folds = []
    c_index_folds = []
    f1_folds = []
    f1_folds_pos = []
    unsuccessful_count = 0
    model_count = 0
    n_manytimes = 8

    dataset, df = construct_graph_dataset(args)
    splitter = CrossValidationSplitter(dataset, df, n=5, n_manytimes=n_manytimes)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.7))
    fold_num=0

    for train_dataset, test_dataset, train_df, test_df in splitter:
        print("starting fold %d-%d" % (fold_num // 5, fold_num % 5))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        train_history = []
        test_history = []
        minimum_loss = None
        auc_fold = None
        acc_fold = None
        early_stop_count = 0
        model = GNN(32)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0004, weight_decay=0.001)

        for epoch in range(n_epoch):
            model.train()
            for data in train_loader:  # Iterate in batches over the training dataset.
                y_pred = model(data.x, data.edge_index, data.batch).view(-1)  # Perform a single forward pass.
                loss = criterion(y_pred, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.

            if epoch % 1 == 0:
                model.eval()
                y_pred_train, y_train = concat_result(train_loader, model)
                y_pred_test, y_test = concat_result(test_loader, model)
                loss_train, loss_test = criterion(y_pred_train, y_train), criterion(y_pred_test, y_test)
                #loss_test = nn.functional.mse_loss(result_test, Y_test)
                acc_train, acc_test = accuracy(y_pred_train, y_train), accuracy(y_pred_test, y_test)
                auc_train, auc_test = auc(y_pred_train, y_train),  auc(y_pred_test, y_test)
                if args.changhai:
                    c_index_train, c_index_test = 0, 0
                else:
                    c_index_train, c_index_test = c_index(y_pred_train, train_df), c_index(y_pred_test, test_df)
                f1_train, f1_test = f1(y_pred_train, y_train, negative = True),  f1(y_pred_test, y_test, negative = True)
                if epoch % 5 == 0:
                    print(f'Epoch:{epoch:03d} Loss:{loss_train:.3f}/{loss_test:.3f} ACC:{acc_train:.3f}/{acc_test:.3f} AUC:{auc_train:.3f}/{auc_test:.3f} CI:{c_index_train:.3f}/{c_index_test:.3f} f1(neg):{f1_train:.3f}/{f1_test:.3f}')

                # early stop
                if minimum_loss is None or minimum_loss * 0.997 > loss_test:
                # if minimum_loss is None or minimum_loss > loss_test:
                    if f1_train == 0:
                        continue
                    minimum_loss = loss_test
                    auc_fold = auc_test
                    acc_fold = acc_test
                    c_index_fold = c_index_test
                    f1_fold = f1_test
                    early_stop_count = 0
                    if acc_fold > 0.75 and auc_fold > 0.75:
                        model.save(args.data_dir + "/model/graph_%d" % model_count)
                #elif auc_test > auc_fold and auc_test>0.5 and acc_test >= acc_fold:
                #    minimum_loss = loss_test
                #    auc_fold = auc_test
                #    acc_fold = acc_test
                #    c_index_fold = c_index_test
                #    f1_fold = f1_test
                #    early_stop_count = 0\
                elif auc_fold + acc_fold + c_index_fold < auc_test + acc_test + c_index_fold:
                    minimum_loss = loss_test
                    auc_fold = auc_test
                    acc_fold = acc_test
                    c_index_fold = c_index_test
                    f1_fold = f1_test
                    early_stop_count = 0
                    if acc_fold > 0.75 and auc_fold > 0.75:
                        model.save(args.data_dir + "/model/graph_%d" % model_count)
                else:
                    early_stop_count += 1
                if auc_fold - 1 < 0.0001:
                    print('wtf')
                if early_stop_count > 3 and epoch>25:
                    if args.stage_two:
                        if auc_fold>0.55 and acc_fold > 0.55:
                            print('early stop at epoch %d' % epoch)
                            if acc_fold > 0.75 and auc_fold > 0.75:
                                model_count += 1
                            break
                    elif early_stop_count > 3:
                        print('early stop at epoch %d' % epoch)
                        break
        else:
            acc_folds.append(acc_fold)
            auc_folds.append(auc_fold)
            f1_folds.append(f1_fold)
            c_index_folds.append(c_index_fold)
        fold_num += 1
        print("acc:%.3f\tauc:%.3f\tc_index:%.3f\tf1:%.3f"  % (acc_fold, auc_fold, c_index_fold, f1_fold))

    total_count =  5 * n_manytimes
    print('CV-acc:%.3f CV-auc:%.3f CV-c-index:%.3f f1(neg):%.3f' % (sum(acc_folds) / total_count, sum(auc_folds) / total_count, sum(c_index_folds)  / total_count,  sum(f1_folds)  / total_count))

if __name__=='__main__':
    main()