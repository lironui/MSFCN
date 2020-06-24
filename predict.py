from __future__ import print_function
import torch
import torch.nn.functional as F
from dataset import train_dataset
from models.Nets3D import FCN3D, FGC3D, UNet3D, MSFCN, MFCN, MSFCNGPM, FSFCN
from models.tiramisu import FCDenseNet57
from models.CeNet import AttU_Net
from measure import SegmentationMetric
import os
from PIL import Image
import numpy as np
import time

cuda = True

path = '2017'
if path == '2017':
    time_series = 7
else:
    time_series = 4

model = FGC3D(time_series, 4, 4)
data_path = './' + path + 'data/test'
test_datatset_ = train_dataset(data_path, time_series=time_series)
#model = Segnet(3,3)
#model_path = './checkpoint/Segnet/model/netG_final.pth'
# model = FGC3DDual(time_series, 4, 4)
model_path = './checkpoint/' + model.name + '/model/' + path + 'netG.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
metric = SegmentationMetric(4)
out_file = './result/' + model.name + '/' + path + '/'

if cuda:
    model.cuda()

try:
    os.makedirs(out_file)
except OSError:
    pass


if __name__ == '__main__':
    start = time.time()
    for i in range(0, test_datatset_.__len__(), 1):
        test_datatset_next = test_datatset_.__next__()
        test_loader = torch.utils.data.DataLoader(dataset=test_datatset_next, batch_size=1, shuffle=True,
                                                   num_workers=0)
        for initial_image, semantic_image in test_loader:
            # print(initial_image.shape)
            initial_image = initial_image.cuda()
            semantic_image = semantic_image.cuda()

            # semantic_image_pred = model(initial_image)
            semantic_image_pred = model(initial_image).detach()
            semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
            semantic_image_pred = semantic_image_pred.argmax(dim=0)

            semantic_image = torch.squeeze(semantic_image.cpu(), 0)
            semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

            metric.addBatch(semantic_image_pred, semantic_image)
            image = semantic_image_pred
            # for i in range(255):
            #     for j in range(255):
            #         if semantic_image[j][i] == 255:
            #             image[j][i] = 255
            # image = Image.fromarray(np.uint8(image))
            # image.save(out_file + str(test_datatset_.get_list()) + ".tif")
    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    oa = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    kappa = metric.kappa()
    aa = metric.meanPixelAccuracy()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print('oa: ', oa)
    print('kappa', kappa)
    print('mIoU: ', mIoU)
    print('aa: ', aa)
    print('FWIoU: ', FWIoU)



