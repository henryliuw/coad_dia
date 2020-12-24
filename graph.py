import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pickle
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.data import DataLoader
import random
from evaluation import accuracy, auc, c_index, recall, f1, precision
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances as euclidean_dist
from module import DatasetLoader, GNN, CrossValidationSplitter

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/new_2000', help='determine the base dir of the dataset document')
    parser.add_argument("--sample_n", default=2000, type=int, help='starting image index of preprocessing')
    parser.add_argument("--evidence_n", default=500, type=int, help='how many tiles to pick from while construction a graph representation')
    parser.add_argument("--batch_size", default=30, type=int, help="batch size")
    parser.add_argument("--threshold", default=25, type=float, help='threshold')
    parser.add_argument("--changhai", action='store_true', help='if use additional data')
    parser.add_argument("--TH", action='store_true')
    args = parser.parse_args()

    gpu = "cuda:2"
    n_epoch = 80
    acc_folds = []
    auc_folds = []
    c_index_folds = []
    f1_folds = []
    f1_folds_pos = []
    unsuccessful_count = 0
    model_count = 0
    n_manytimes = 2

    # caching
    dataset, df = construct_graph_dataset(args, gpu)
    '''if True:
    # if os.path.exists(os.path.join(args.data_dir, 'graph', 'graph_dataset.pkl')) and os.path.exists(os.path.join(args.data_dir, 'graph', 'graph_df.pkl')):
        print("loading cached graph data")
        with open(os.path.join(args.data_dir, 'graph', 'graph_dataset.pkl'), 'rb') as file:
            dataset = pickle.load(file)
        with open(os.path.join(args.data_dir, 'graph', 'graph_df.pkl'), 'rb') as file:
            df = pickle.load(file)
    else:
        if not os.path.exists(os.path.join(args.data_dir, 'graph')):
            os.mkdir(os.path.join(args.data_dir, 'graph'))
        dataset, df = construct_graph_dataset(args, gpu)
        with open(os.path.join(args.data_dir, 'graph', 'graph_dataset.pkl'), 'wb') as file:
            pickle.dump(dataset, file)
        with open(os.path.join(args.data_dir, 'graph', 'graph_df.pkl'), 'wb') as file:
            pickle.dump(df, file)
    '''

    splitter = CrossValidationSplitter(dataset, df, n=5, n_manytimes=n_manytimes)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.6))
    #criterion = nn.BCELoss(pos_weight=torch.tensor(0.8))
    fold_num=0
    if not os.path.isdir(os.path.join(args.data_dir, 'model')):
        os.mkdir(os.path.join(args.data_dir, 'model'))

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
        model = GNN(32).cuda()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0004, weight_decay=0.001)

        for epoch in range(n_epoch):
            model.train()
            for data in train_loader:  # Iterate in batches over the training dataset.
                y_pred = model(data.x, data.edge_index, data.batch.cuda()).view(-1)  # Perform a single forward pass.
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
                y_pred_test = torch.nn.functional.sigmoid(y_pred_test)
                y_pred_train = torch.nn.functional.sigmoid(y_pred_train)
                acc_train, acc_test = accuracy(y_pred_train, y_train), accuracy(y_pred_test, y_test)
                auc_train, auc_test = auc(y_pred_train, y_train),  auc(y_pred_test, y_test)
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
                if abs(auc_fold - 1) < 0.0001:
                    pass
                    #print('wtf')
                if early_stop_count > 3 and epoch>25:
                    if auc_fold>0.6 and acc_fold > 0.6:
                        print('early stop at epoch %d' % epoch)
                        if acc_fold > 0.75 and auc_fold > 0.75:
                            model.load(args.data_dir + "/model/graph_%d" % model_count)
                            model_count += 1
                        break
        
        acc_folds.append(acc_fold)
        auc_folds.append(auc_fold)
        f1_folds.append(f1_fold)
        c_index_folds.append(c_index_fold)
        fold_num += 1
        print("acc:%.3f\tauc:%.3f\tc_index:%.3f\tf1:%.3f"  % (acc_fold, auc_fold, c_index_fold, f1_fold))

    total_count = 5 * n_manytimes
    print('CV-acc:%.3f CV-auc:%.3f CV-c-index:%.3f f1(neg):%.3f' % (sum(acc_folds) / total_count, sum(auc_folds) / total_count, sum(c_index_folds)  / total_count,  sum(f1_folds)  / total_count))


def to_PyG_graph(data, loc_file, y, model='None', gpu=None, threshold=80, evidence_n=20, method='random',return_edge=False):
    
    if method=='score':
        evidence_n *= 2
        score = model.tile_scoring(data.reshape(1,32,2000)).view(2000).detach()
        bot_idx = np.argsort(score)[-evidence_n:]
        top_idx = np.argsort(score)[:evidence_n]
        all_idx = np.r_[bot_idx, top_idx]
    elif method=='random':
        all_idx = random.sample(range(2000), evidence_n * 2)

    with open(loc_file, 'rb') as file:
        location = pickle.load(file)
    location_mat = np.array(location)[all_idx]

    dist_1 = location_mat[:, 1].reshape(-1,1) - location_mat[:,1].reshape(1,-1)
    dist_0 = location_mat[:, 0].reshape(-1,1) - location_mat[:,0].reshape(1,-1)
    dist = ((dist_0 ** 2 + dist_1 ** 2) ** 0.5)
    A = np.zeros_like(dist, dtype=np.int)
    A [(dist < threshold)] = 1
    A[range(evidence_n),range(evidence_n)] = 0
    # print("Average node degree:%.2f" % (A.sum() / 2.0 /evidence_n))

    edge_index = []
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j]:
                edge_index.append((i,j))
    edge_index = torch.tensor(edge_index).t().contiguous().cuda()
    
    x = data[:, all_idx].t().contiguous().cuda()
    # x = score[all_idx].reshape(2 * evidence_n, 1)
    graph_data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float).cuda()) 

    if return_edge:

        return graph_data, location_mat, edge_index.numpy().T
    else:
        return graph_data

def get_original_file(loc_file, type):
    if type=="tcga":
        useful_subset = pd.read_csv('data/useful_subset.csv')
        i = int(loc_file.split('/')[-1].split('_')[0])
        # that_row = dataset.df[dataset.df['index']==i]
        image_path = os.path.join('/home/DiskB/tcga_coad_dia', useful_subset.loc[i, 'id'],useful_subset.loc[i, 'File me'])
    elif type=="changhai":
        i = int(loc_file.split('/')[-1].split('_')[0])
        image_path = os.path.join('/home/DiskB/tcga_coad_dia/changhai', i+'.svs')
    return image_path

def plot_graph(loc_file, aggregated_location, edge_index, type='tcga'):
    original_file = get_original_file(loc_file, type)
    slide = openslide.OpenSlide(original_file)
    level_downsamples = sorted([round(i) for i in slide.level_downsamples], reverse=True)
    low_resolution_img = np.array(slide.read_region((0,0), len(slide.level_dimensions)-1, slide.level_dimensions[-1]))
    plt.figure(figsize=(15, 15))
    plt.imshow(low_resolution_img[:,:,:3])
    for (i, j) in edge_index:
        plt.plot([aggregated_location[i][0] * 7, aggregated_location[j][0]*7], [aggregated_location[i][1]*7, aggregated_location[j][1]*7], c='black')
    
    plt.scatter(aggregated_location[:,0] * 7, aggregated_location[:,1]*7)

'''
def to_cluster_graph(data, image_file, y, gpu=None, threshold=20, evidence_n=200):
    #clustering
    data = data.numpy().T
    with open(image_file, 'rb') as file:
        location = pickle.load(file)
    spatial_dist = euclidean_dist(location)
    feature_dist = euclidean_dist(data)
    combined_dist = feature_dist * spatial_dist
    kmeans = KMeans(n_clusters=evidence_n).fit(combined_dist)
    
    # aggregate
    aggregated_location = []
    aggregated_feature = []
    for i in range(evidence_n):
        idx = np.where(kmeans.labels_==i)[0]
        nodes_location = np.array(location)[idx]
        nodes_feature = data[idx]
        aggregated_location.append(nodes_location.mean(axis=0))
        aggregated_feature.append(nodes_feature.mean(axis=0))
    
    aggregated_location = np.array(aggregated_location)
    aggregated_feature = np.array(aggregated_feature)

    dist = euclidean_dist(aggregated_location)
    A = np.zeros_like(dist, dtype=np.int)
    A [(dist < threshold)] = 1
    A[range(evidence_n),range(evidence_n)] = 0

    edge_index = []
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j]:
                edge_index.append((i,j))
    edge_index = torch.tensor(edge_index).t().contiguous().cuda()
    
    x = torch.tensor(aggregated_feature).cuda()
    # x = score[all_idx].reshape(2 * evidence_n, 1)
    graph_data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float).cuda()) 

    return graph_data
    #return graph_data, aggregated_location, edge_index.numpy().T
'''

def construct_graph_dataset(args, gpu):
    dataloader = DatasetLoader(args, gpu=gpu, feature_size=32)
    dataloader.df.reset_index(inplace=True)
    #model = Predictor(evidence_size=20, layers=(100, 50, 1), feature_size=32)
    #model.load(args.data_dir+'/model/model_0')
    dataset = []
    for i in dataloader.df.index:
        print("\r","constructing graph representation %d/%d" % (i, len(dataloader.df)) , end='', flush=True)
        data = dataloader.X[i]
        loc_file =  dataloader.df.loc[i, 'loc_file']
        # dataset.append(to_PyG_graph(data, image_file, dataloader.df.loc[i, 'y2'], model, gpu, threshold=args.threshold, evidence_n=args.evidence_n, method='random'))
        dataset.append(to_PyG_graph(data, loc_file, dataloader.df.loc[i, 'y2'], None, gpu, args.threshold, args.evidence_n))
    print("")
    return dataset, dataloader.df

def concat_result(dataloader, model, gpu=True):
    out = []
    target = []
    for data in dataloader:
        out.append(model(data.x, data.edge_index, data.batch.cuda()).view(-1))
        target.append(data.y)
    return torch.cat(out), torch.cat(target)


if __name__=='__main__':
    main()