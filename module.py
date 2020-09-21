import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
import lifelines
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore') # possibly harmful code

class Predictor(nn.Module):
    def __init__(self, evidence_size=5, layers=[200, 100, 1]):
        # tile scoring
        super().__init__()
        self.conv1d_layer = torch.nn.Conv1d(2048, 1, 1)
        self.evidence_size=evidence_size #R
        linear_layer_list = []
        for i in range(len(layers)):
            if i==0:
                linear_layer_list.append(nn.Linear(2 * self.evidence_size, layers[i]))
            else:
                linear_layer_list.append(nn.Linear(layers[i-1], layers[i]))
            if (i+1)!=len(layers):
                linear_layer_list.append(nn.ReLU(inplace=True))
        self.linear_layers = nn.Sequential(*linear_layer_list)
        self.sigmoid_f = torch.nn.Sigmoid()
    def forward(self, image_features):
        # input [batch, 2048, Tile_size]
        tile_descriptor = self.conv1d_layer(image_features)
        # [batch, 1, Tile_size]
        top_items, _ = torch.topk(tile_descriptor, self.evidence_size, dim=2) 
        bottom_items, _ = torch.topk(tile_descriptor, self.evidence_size, dim=2, largest=False)
        evidences = torch.cat((top_items, bottom_items), dim=2).view(-1, self.evidence_size * 2)
        # items [batch, 2 * evidence_size]
        logits = self.linear_layers(evidences) 
        results = self.sigmoid_f(logits).view(-1)
        return results

    def tile_scoring(self, image_features):
        return self.conv1d_layer(image_features)

    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir+'/model')

    def load(self, save_dir):
        self.load_state_dict(torch.load(save_dir+'/model'))

def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias: 
            torch.nn.init.kaiming_uniform_(m.bias)

class CVDataLoader():
    ''' a 5-folds cross validation dataloader, with '''
    def __init__(self, args, gpu):
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

        self.gpu = gpu
        self.X, self.Y = torch.Tensor(X), torch.Tensor(Y)
        self.df = df_new
        self.repl_n = args.repl_n
        self.image_split = args.image_split

    def _init_fold(self, X, Y, df, repl_n, gpu=True, image_split=True):


        if image_split:
            # train test will not have overlapping datapoints from a same image
            img_idx = df['sample_id'].unique()
            np.random.shuffle(img_idx)
            id1, id2, id3, id4 = int(0.2 * len(img_idx)), int(0.4 * len(img_idx)), int(0.6 * len(img_idx)), int(0.8 * len(img_idx))
            self.perm_idx = np.array((img_idx[:id1], img_idx[id1:id2], img_idx[id2:id3], img_idx[id3:id4], img_idx[id4:]))
        else:
            perm = np.random.permutation(len(Y))
            id1, id2, id3, id4 = int(0.2 * len(Y)), int(0.4 * len(Y)), int(0.6 * len(Y)), int(0.8 * len(Y))
            self.perm_idx = np.array((perm[:id1], perm[id1:id2], perm[id2:id3], perm[id3:id4], perm[id4:]))
    
    def set_fold(self, i):
        train_idx_slice = [_ for _ in range(5)]
        train_idx_slice.remove(i)
        if self.image_split:
            self.train_idx = self.df['sample_id'].isin(np.concatenate(self.perm_idx[train_idx_slice]))
            self.test_idx = self.df['sample_id'].isin(self.perm_idx[i])
        else:
            self.train_idx = np.concatenate(self.perm_idx[train_idx_slice])
            self.test_idx = self.perm_idx[i]

        self.idx = [int(len(self.train_idx) * i / self.repl_n) for i in range(self.repl_n)] + [len(self.train_idx)] # for batch
        self.batch_i = 0
        self.X_train = self.X[self.train_idx]
        self.Y_train = self.Y[self.train_idx]
        if self.image_split:
            self.df_train = self.df[self.train_idx]
            self.df_test = self.df[self.test_idx]
        else:
            self.df_train = self.df.iloc[self.train_idx]
            self.df_test = self.df.iloc[self.test_idx]

        self.X_test = self.X[self.test_idx]
        self.Y_test = self.Y[self.test_idx]

        if self.gpu:
            self.X_test = self.X_test.cuda()
            self.Y_test = self.Y_test.cuda()

    def get_test(self):
        return self.X_test, self.Y_test, self.df_test
    
    def get_train(self):
        if self.gpu:
            return self.X_train.cuda(), self.Y_train.cuda(), self.df_train
        else:
            return self.X_train, self.Y_train, self.df_train

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_i == self.repl_n:
            raise StopIteration
        else:
            train_X_batch = self.X_train[self.idx[self.batch_i]:self.idx[self.batch_i + 1]]
            train_Y_batch = self.Y_train[self.idx[self.batch_i]:self.idx[self.batch_i + 1]]
            train_df_batch = self.df_train[self.idx[self.batch_i]:self.idx[self.batch_i + 1]]
            self.batch_i += 1
            if self.gpu:
                train_X_batch = train_X_batch.cuda()
                train_Y_batch = train_Y_batch.cuda()
            return train_X_batch, train_Y_batch, train_df_batch

    def a
    
def accuracy(result, target): 
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return torch.eq(result > 0.5, target.type(torch.BoolTensor)).sum().numpy() / len(target)

def auc(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return roc_auc_score(target.numpy(), result.detach().numpy())

def c_index(result, df):
    if result.is_cuda:
        result = result.cpu()
    #try:
    #    cph = CoxPHFitter().fit(df, duration_col="time", event_col="y", formula="predict")
    #except:
    #    return -1
    #return cph.concordance_index_
    return lifelines.utils.concordance_index(df['time'], result.detach().numpy(), ~df['y'])