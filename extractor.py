''' extractor all in one '''

import numpy as np
import torch, torchvision
import gc
import time
import random
from module import accuracy, auc

# tumor mean array([182.04832497, 134.43371082, 161.65735417])
# tumor std array([46.54416987, 51.39264384, 38.47625945])
# normal mean array([180.81651856, 140.7951604 , 166.7642652 ])
# normal std array([51.42186896982495,60.400144076706326, 45.423789700643006])
# total mean array([181.12447016, 139.204798  , 165.48753744])
# total std array([50.20244419, 58.14826902, 43.68690714])

def main():
    # training 
    batch_size = 100
    epoch_n = 500
    lr = 0.0005
    weight_decay = 0.001

    print("%s loading data" % time.strftime('%m.%d %H:%M:%S'))
    dataloader = CAMdataloader(batch_size)
    model = MyResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_test, Y_test = dataloader.get_test()

    print("%s training start" % time.strftime('%m.%d %H:%M:%S'))
    for epoch in range(epoch_n):
        for X_batch, Y_batch in dataloader:
            Y_predict = torch.nn.functional.sigmoid(model(X_batch))
            loss = torch.nn.functional.binary_cross_entropy(Y_predict, Y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # evaluation
            if epoch % 5 == 0:
                Y_predict_test = model(X_test)
                loss_test = nn.functional.binary_cross_entropy(Y_predict_test, Y_test)
                acc_train, acc_test = accuracy(Y_predict, Y_batch), accuracy(Y_predict_test, Y_test)
                auc_train, auc_test = auc(Y_predict, Y_batch),  auc(Y_predict_test, Y_test)
                print("%s epoch:%d loss:%.3f acc:%.3f auc:%.3f test_loss:%.3f test_acc:%.3f test_auc:%.3f" % 
                        (time.strftime('%m.%d %H:%M:%S', time.localtime(time.time())), epoch, loss, acc_train, auc_train,  loss_test, acc_test, auc_test))

class MyResNet(torchvision.models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(num_classes=1, *args, **kwargs)
    
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
        return x

    def save(self, name=""):
        torch.save(self.state_dict(), 'data/cam/model_%s' % name)

    def load(self, name=""):
        self.load_state_dict(torch.load('data/cam/model_%s') % name)

class CAMdataloader():
    def __init__(self, batch_size=100):
        # self.gpu = True
        assert 1000 % batch_size == 0, "batch size undivisable by 1000 is unsupported now :("
        self.batch_size = batch_size
        self.test_idx = random.sample(range(100), 5)
        self.X_test = torch.tensor(np.r_[[np.load('data/cam/X_1000_%d.npy' % i) for i in self.test_idx]], dtype=torch.float).cuda()
        self.Y_test = torch.tensor(np.r_[[np.load('data/cam/Y_1000_%d.npy' % i) for i in self.test_idx]], dtype=torch.float).cuda()
        
    def __iter__(self):
        self.i_1000 = 0
        while self.i_1000 in self.test_idx:
            self.i_1000 += 1
        self.i = 0
        self.i_len = 1000 // self.batch_size
        self.X_1000 = torch.tensor(np.load('data/cam/X_1000_%d.npy' % self.i_1000), dtype=torch.float).cuda()
        self.Y_1000 = torch.tensor(np.load('data/cam/Y_1000_%d.npy' % self.i_1000), dtype=torch.float).cuda()
        return self
    
    def __next__(self):
        if self.i == self.i_len: # read new 1000 patch
            self.i_1000 += 1
            while self.i_1000 in self.test_idx:
                self.i_1000 += 1
            if self.i_1000 == 100:
                raise StopIteration
            self.i = 0
            self.X_1000 = torch.tensor(np.load('data/cam/X_1000_%d.npy' % self.i_1000), dtype=torch.float).cuda()
            self.Y_1000 = torc.tensor(np.load('data/cam/Y_1000_%d.npy' % self.i_1000), dtype=torch.float).cuda()
        # iterate over patch to get minibatch
        return self.X_1000[self.batch_size * self.i: self.batch_size * (self.i+1)], self.Y_1000[self.batch_size * self.i: self.batch_size * (self.i+1)]

    def get_test(self):
        return self.X_test, self.Y_test

    

if __name__ == '__main__':
    main()