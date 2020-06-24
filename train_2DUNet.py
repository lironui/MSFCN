from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from dataset import train_dataset
from models.Nets2D import UNet2D
from early_stopping import EarlyStopping
from measure import SegmentationMetric

path = '2017'
if path == '2017':
    time_series = 7
else:
    time_series = 4
batch_size = 1
niter = 100
input_channel = 4
class_num = 4
learning_rate = 0.0001
beta1 = 0.5
cuda = True
num_workers = 1
size_h = 256
size_w = 256
flip = 0
net = UNet2D(time_series, 4, 4)
data_path = './' + path + 'data/train'
val_path = './' + path + 'data/val'
out_file = './checkpoint/' + net.name
save_epoch = 1
test_step = 300
log_step = 1
num_GPU = 1
pre_trained = False

torch.cuda.set_device(0)

try:
    os.makedirs(out_file)
    os.makedirs(out_file + '/model/')
except OSError:
    pass

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_datatset_ = train_dataset(data_path, size_w, size_h, flip, time_series)
val_datatset_ = train_dataset(val_path, size_w, size_h, 0, time_series)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

try:
    os.makedirs(out_file)
    os.makedirs(out_file + '/model/')
except OSError:
    pass
if cuda:
    net.cuda()
if num_GPU > 1:
    net = nn.DataParallel(net)

if pre_trained:
    net.load_state_dict(torch.load('%s/model/' % out_file + path + 'netG.pth'))
    # print('Load success!')
else:
    pass
    # net.apply(weights_init)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
metric = SegmentationMetric(4)
early_stopping = EarlyStopping(patience=7, verbose=True)

if __name__ == '__main__':
    start = time.time()
    net.train()
    for epoch in range(1, niter+1):
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.0, last_epoch=-1)
        for i in range(0, train_datatset_.__len__(), batch_size):
            train_datatset_next = train_datatset_.__next__()
            train_loader = torch.utils.data.DataLoader(dataset=train_datatset_next, batch_size=batch_size, shuffle=True,
                                                       num_workers=num_workers)
            for initial_image, semantic_image in train_loader:
                # print(initial_image.shape)
                initial_image = torch.reshape(initial_image, (batch_size, time_series*4, 256, 256))
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image)

                loss = criterion(semantic_image_pred, semantic_image.long())
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_adjust.step()
            print('[%d/%d][%d/%d] Loss: %.4f' %
                    (epoch, niter, i, len(train_loader) * batch_size, loss.item()))

        for i in range(0, val_datatset_.__len__(), batch_size):
            with torch.no_grad():
                net.eval()
                val_datatset_next = val_datatset_.__next__()
                val_loader = torch.utils.data.DataLoader(dataset=val_datatset_next, batch_size=batch_size, shuffle=True,
                                                           num_workers=num_workers)
                for initial_image, semantic_image in val_loader:
                    # print(initial_image.shape)
                    initial_image = torch.reshape(initial_image, (batch_size, time_series * 4, 256, 256))
                    initial_image = initial_image.cuda()
                    semantic_image = semantic_image.cuda()

                    semantic_image_pred = net(initial_image)

                    loss = criterion(semantic_image_pred, semantic_image.long())
                    semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                    semantic_image_pred = semantic_image_pred.argmax(dim=0)

                    semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                    semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                    metric.addBatch(semantic_image_pred, semantic_image)
        acc = metric.pixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        kappa = metric.kappa()
        print('acc: ', acc)
        print('mIoU: ', mIoU)
        print('kappa', kappa)
        metric.reset()
        net.train()

        early_stopping(1 - mIoU, net, '%s/model/' % out_file + path + 'netG.pth')

        if early_stopping.early_stop:
            break

    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')