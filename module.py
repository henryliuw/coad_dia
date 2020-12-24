import torch.nn as nn
import torch
import numpy as np
import os
import pandas as pd
import warnings
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from torch_geometric.nn import GCNConv, GraphConv, GENConv, GINConv
from torch_geometric.nn import global_mean_pool, global_sort_pool, Set2Set, GlobalAttention, global_add_pool

warnings.filterwarnings('ignore') # possibly harmful code


class GNN(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        torch.manual_seed(12345)
        #self.conv1 = GraphConv(feature_size, 32)
        #self.bn1 = nn.BatchNorm1d(32)
        #self.conv2 = GraphConv(32, 32)
        #self.bn2 = nn.BatchNorm1d(32)
        #self.conv3 = GraphConv(32, 16)
        #self.bn3 = nn.BatchNorm1d(16)
        #self.conv4 = GraphConv(16, 16)
        #self.bn4 = nn.BatchNorm1d(16)
        self.conv1 = GINConv(nn.Sequential(nn.Linear(feature_size,32),  nn.BatchNorm1d(32), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(32,32)), train_eps=True)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32,64),   nn.BatchNorm1d(64), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(64,32)), train_eps=True)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(32,64),   nn.BatchNorm1d(64), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(64,32)), train_eps=True)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(32,64),  nn.BatchNorm1d(64), nn.Dropout(p=0.5),  nn.ReLU(), nn.Linear(64,16)), train_eps=True)
        # self.conv5 = GINConv(nn.Sequential(Linear(16,64),  nn.BatchNorm1d(64), nn.Dropout(p=0.2), nn.ReLU(), Linear(64,16)), train_eps=True)
        # self.readout = Set2Set(8, 5)
        # self.mlp = torch.nn.Sequential(Linear(8,8), F.Re(), Linear(8,1))
        #self.readout = GlobalAttention(torch.nn.Sequential(nn.Linear(8,1)))
        self.readout = global_mean_pool
        self.lin = nn.Linear(16, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        #x = self.bn1(x)
        x = self.conv2(x, edge_index)
        #x = self.bn2(x)
        x = self.conv3(x, edge_index)
        #x = self.bn3(x)
        x = self.conv4(x, edge_index)
        #x = self.bn4(x)
        #x = self.conv5(x, edge_index)
        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.readout(x, batch)
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        #x = torch.nn.functional.sigmoid(x)
        return x

    def save(self, save_file):
        torch.save(self.state_dict(), save_file)

    def load(self, save_file):
        self.load_state_dict(torch.load(save_file))

class DatasetLoader():
    ''' a 5-folds cross validation dataloader, with '''
    def __init__(self, args, gpu, feature_size=2048):
        # reading in preprocessed data
        X_cached_file = os.path.join(args.data_dir, 'X_%d.npy' % args.sample_n)
        df_cached_file = os.path.join(args.data_dir, 'df.csv')
        
        if os.path.exists(X_cached_file) and os.path.exists(df_cached_file):
            print('loading cached data')
            with open(X_cached_file,'rb') as file:
                X = np.load(file)
            df_new = pd.read_csv(df_cached_file)
        else:
            X, Y = None, None
            df_new = None
            print('reading data from preprocessed file')
            csv_file = 'data/useful_subset.csv'
            useful_subset = pd.read_csv(csv_file)
            useful_subset['OS.time'].fillna(useful_subset['OS.time'].mean(), inplace=True)
            size = len(useful_subset)
            for i in range(0, size):
                if not useful_subset.loc[i, "stage"] == 'Stage II':
                    continue
                X_this_file = args.data_dir+'/tcga/'+str(i)+'_features.npy'
                X_this = np.load(X_this_file, allow_pickle=True)
                Y_this = useful_subset.loc[i, 'outcome'] == 'good'
                Y_this_stage_two = useful_subset.loc[i, 'outcome2'] == 'good'
                if len(X_this)==args.sample_n:
                    if X is None:
                        X = X_this.reshape(1, args.sample_n, feature_size)
                    else:
                        X = np.r_[X, X_this.reshape(1, args.sample_n, feature_size)]
                    loc_file = args.data_dir+'/tcga/'+str(i)+'_name.pkl'
                    image_path = os.path.join('/home/DiskB/tcga_coad_dia', useful_subset.loc[i, 'id'], useful_subset.loc[i, 'File me'])
                    if df_new is None:
                        df_new = pd.DataFrame({"y": Y_this, "y2": Y_this_stage_two, "time": useful_subset.loc[i, "OS.time"], 'sample_id':i, 'image_file':image_path, 'loc_file':loc_file, 'stage_two':True, "source":"tcga"}, index=[0])
                    else:
                        df_new = df_new.append({"y": Y_this, "y2": Y_this_stage_two, "time": useful_subset.loc[i, "OS.time"], 'sample_id':i, 'image_file':image_path, 'loc_file':loc_file, 'stage_two':True, "source":"tcga"}, ignore_index=True)
                print("\r","reading data input  %d/%d from TCGA" % (i, size) , end='', flush=True)
            print("")

            # read in changhai data
            data_xls = pd.ExcelFile('data/information.xlsx')
            df_changhai = data_xls.parse(sheet_name='changhai')
            size = len(df_changhai)
            X_changhai = []
            for i in df_changhai.index:
                if df_changhai.loc[i, 'use']:
                    image_file = os.path.join('/home/DiskB/COAD_additional_data/changhai', str(df_changhai.loc[i, 'filename'])+'.svs')
                    file_dir = os.path.join(args.data_dir, 'changhai', '%d_features.npy' % i)
                    if not os.path.exists(file_dir):
                        continue
                    X_changhai.append(np.load(file_dir).reshape(1,2000,32))
                    y = (df_changhai.loc[i, 'outcome'] == 'well')
                    loc_file = file_dir.strip('features.npy') + 'name.pkl'
                    df_new = df_new.append({"y": y, "y2": y, "time": df_changhai.loc[i, 'OS.time'], 'sample_id': i, 'image_file':image_file, 'loc_file':loc_file, 'stage_two':True, "source":"changhai"}, ignore_index=True)
                print("\r","reading data input  %d/%d from changhai" % (i, size) , end='', flush=True)
            X_changhai = np.concatenate(X_changhai)
            print("")

            # read in TU data
            df_th = data_xls.parse(sheet_name='TumorHospital')
            X_th = []
            size = len(df_th)
            for i in df_th.index:
                image_files = os.listdir('/home/DiskB/COAD_additional_data/TumorHospital')
                if df_th.loc[i, 'use']:
                    file_id = df_th.loc[i, 'filename'][2:]
                    for image_file in image_files:
                        if file_id in image_file:
                            file_dir =  os.path.join(args.data_dir, 'TH', "%s_features.npy" % file_id)
                            loc_file = file_dir.strip('features.npy') + 'name.pkl'
                            if os.path.exists(file_dir):
                                X_th.append(np.load(file_dir).reshape(1,2000,32))
                                y = (df_th.loc[i, 'outcome'] == 'well')
                                df_new = df_new.append({"y": y, "y2": y, "time": df_th.loc[i, 'OS.time'], 'sample_id': file_id, 'image_file':image_file, 'loc_file':loc_file, 'stage_two':True, "source":"TH"}, ignore_index=True)
                print("\r","reading data input  %d/%d from TumorHospital" % (i, size) , end='', flush=True)
            print("")
            X_th = np.concatenate(X_th)

            X = np.r_[X, X_changhai, X_th]
            X = X.transpose((0, 2, 1))
            try:
                with open(X_cached_file,'wb') as file:
                    np.save(file, X)
                df_new.to_csv(df_cached_file)
            except Exception as e:
                print(e)
            
        # retrieve X, Y
        # always use stage_two


        data_source = ['tcga']
        if args.changhai:
            data_source.append('changhai')
        if args.TH:
            data_source.append('TH')

        idx = df_new['source'].isin(data_source)
        X = X[idx]
        Y = df_new[idx]["y2"].values
        df_new = df_new[idx]
        
        # the first data in X (tcga) is deleted since has the same patient as the second data
        X = X[2:]
        df_new = df_new[2:]
        Y = Y[2:]

        self.gpu = gpu
        self.X, self.Y = torch.Tensor(X), torch.Tensor(Y)
        self.df = df_new
        # self.batch_size = args.batch_size

class CrossValidationSplitter():
    def __init__(self, dataset, df, n=5, n_manytimes=8):
        ''' 
        dataset could be a list of datapoints
        '''
        self.n = n
        self.df = df
        self.dataset = dataset
        self.splitter = RepeatedStratifiedKFold(n_splits=self.n, n_repeats=n_manytimes, random_state=random.randint(0,1000))
        self.train_idx = []
        self.test_idx = []
        length_Y = len(self.dataset)
        for train_idx, test_idx in self.splitter.split(np.zeros(length_Y), self.df['y2'].to_numpy()):
            self.train_idx.append(train_idx)
            self.test_idx.append(test_idx)
        self.dataset_df = pd.DataFrame({'data':self.dataset})
        self.n_manytimes = n_manytimes
    def __iter__(self):
        self.n_count = 0
        #split
        #id1, id2, id3, id4 = int(0.2 * length_Y), int(0.4 * length_Y), int(0.6 * length_Y), int(0.8 * length_Y)
        #self.perm_idx = np.array((perm[:id1], perm[id1:id2], perm[id2:id3], perm[id3:id4], perm[id4:]))
        #fold_num = np.zeros(length_Y, dtype=np.int)
        #for i in range(self.n):
        #    fold_num[self.perm_idx[i]] = i
        return self

    def __next__(self):
        if self.n_count == self.n * self.n_manytimes:
            raise StopIteration
        #train_idx_slice = [_ for _ in range(self.n)]
        #train_idx_slice.remove(self.n_count)
        #train_idx = self.dataset_df.fold.isin(train_idx_slice)
        #test_idx = ~train_idx
        train_idx = self.dataset_df.index.isin(self.train_idx[self.n_count])
        test_idx = ~train_idx
        self.n_count += 1
        return self.dataset_df[train_idx].data.values.tolist(), self.dataset_df[test_idx].data.values.tolist(), self.df[train_idx], self.df[test_idx]
