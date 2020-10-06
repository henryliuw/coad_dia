''' extractor all in one '''

import numpy as np
import torch, torchvision
import gc
import time
import random
from module import accuracy, auc
import logging

gpu = "cuda:0"
# tumor mean array([182.04832497, 134.43371082, 161.65735417])
# tumor std array([46.54416987, 51.39264384, 38.47625945])
# normal mean array([180.81651856, 140.7951604 , 166.7642652 ])
# normal std array([51.42186896982495,60.400144076706326, 45.423789700643006])
# total mean array([181.12447016, 139.204798  , 165.48753744])
# total std array([50.20244419, 58.14826902, 43.68690714])

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="", help='save file name')
    args = parser.parse_args()
    # training 
    batch_size = 100
    epoch_n = 50
    lr = 0.0005
    weight_decay = 0.00001
    logging.basicConfig(filename='data/cam/log_' + args.name ,level=logging.INFO)

    msg = "%s loading data" % time.strftime('%m.%d %H:%M:%S')
    print(msg)
    logging.info(msg)

    dataloader = CAMdataloader(batch_size)
    model = MyResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3]).to(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    _, Y_test = dataloader.get_test()
    Y_test = Y_test.to(gpu)

    msg = "%s training start" % time.strftime('%m.%d %H:%M:%S')
    print(msg)
    logging.info(msg)

    for epoch in range(epoch_n):
        total_data = 0
        for X_batch, Y_batch in dataloader:
            total_data += len(X_batch)
            X_batch, Y_batch = X_batch.to(gpu), Y_batch.to(gpu)
            Y_predict = torch.nn.functional.sigmoid(model(X_batch)).view(-1)
            loss = torch.nn.functional.binary_cross_entropy(Y_predict, Y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # evaluation
            if total_data % 10000 == 0:
                Y_predict_test = []
                model.eval()
                for X_batch_test, _ in dataloader.test:
                    X_batch_test = X_batch_test.to(gpu)
                    y_predict_test = torch.nn.functional.sigmoid(model(X_batch_test).detach())
                    Y_predict_test.append(y_predict_test)
                    # torch.cuda.empty_cache()
                Y_predict_test = torch.cat(Y_predict_test).view(-1)

                loss_test = torch.nn.functional.binary_cross_entropy(Y_predict_test, Y_test)
                acc_train, acc_test = accuracy(Y_predict, Y_batch), accuracy(Y_predict_test, Y_test)
                auc_train, auc_test = auc(Y_predict, Y_batch), auc(Y_predict_test, Y_test)
                msg = "%s epoch:%d(%d/100000) loss:%.3f acc:%.3f auc:%.3f test_loss:%.3f test_acc:%.3f test_auc:%.3f" % (time.strftime('%m.%d %H:%M:%S', time.localtime(time.time())), epoch, total_data, loss, acc_train, auc_train, loss_test, acc_test, auc_test)
                print(msg)
                logging.info(msg)
                model.train()
    
    model.save(args.name)

class MyResNet(torchvision.models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(num_classes=1, *args, **kwargs)
        feature_size=32
        self.fc = torch.nn.Linear(512, feature_size)
        self.fc2 = torch.nn.Linear(feature_size, 1)
    def forward(self, x):
        x = self.get_feature(x)
        x = self.fc2(x)
        return x
    def get_feature(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def save(self, name=""):
        torch.save(self.state_dict(), 'data/cam/model_%s' % name)

    def load(self, name=""):
        self.load_state_dict(torch.load('data/cam/model_%s' % name))

class CAMdataloader():
    def __init__(self, batch_size=100):
        # self.gpu = True
        assert 1000 % batch_size == 0, "batch size undivisable by 1000 is unsupported now :("
        self.batch_size = batch_size
        self.test_idx = random.sample(range(100), 2)
        self.X_test = torch.tensor(np.concatenate([np.load('data/cam/X_1000_%d.npy' % i) for i in self.test_idx]), dtype=torch.float)
        self.Y_test = torch.tensor(np.concatenate([np.load('data/cam/Y_1000_%d.npy' % i) for i in self.test_idx]), dtype=torch.float)
        self.test = DataIterator(batch_size, self.X_test, self.Y_test)
    def __iter__(self):
        self.i_1000 = 0
        while self.i_1000 in self.test_idx:
            self.i_1000 += 1
        self.i = 0
        self.i_len = 1000 // self.batch_size
        self.X_1000 = torch.tensor(np.load('data/cam/X_1000_%d.npy' % self.i_1000), dtype=torch.float)
        self.Y_1000 = torch.tensor(np.load('data/cam/Y_1000_%d.npy' % self.i_1000), dtype=torch.float)
        return self
    
    def __next__(self):
        if self.i == self.i_len: # read new 1000 patch
            self.i_1000 += 1
            while self.i_1000 in self.test_idx:
                self.i_1000 += 1
            if self.i_1000 == 100:
                raise StopIteration
            self.i = 0
            self.X_1000 = torch.tensor(np.load('data/cam/X_1000_%d.npy' % self.i_1000), dtype=torch.float)
            self.Y_1000 = torch.tensor(np.load('data/cam/Y_1000_%d.npy' % self.i_1000), dtype=torch.float)
        self.i += 1
        # iterate over patch to get minibatch
        return self.X_1000[self.batch_size * (self.i-1): self.batch_size * self.i], self.Y_1000[self.batch_size * (self.i-1): self.batch_size * self.i]

    def get_test(self):
        return self.X_test, self.Y_test

class DataIterator():
    def __init__(self, batch_size=100, *args):
        self.data = args
        self.batch_size = batch_size
        self.i_len = len(self.data[0]) // self.batch_size
        self.divisable = len(self.data[0]) % self.batch_size == 0
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i == self.i_len:
            if self.divisable:
                raise StopIteration
            else:
                self.i += 1
                return [item[(self.i-1) * self.batch_size:] for item in self.data]
        elif self.i == (self.i_len + 1):
            raise StopIteration
        else:
            self.i += 1
            return [item[(self.i-1) * self.batch_size:self.i * self.batch_size] for item in self.data]


if __name__ == '__main__':
    main()