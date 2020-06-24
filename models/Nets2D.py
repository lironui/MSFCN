from torch import nn
import torch
import torchsummary
from thop import profile


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


class DualConvLayer2D(nn.Module):
    # 3x3x3 convolution with padding and relu
    def __init__(self, in_planes, out_planes):
        super(DualConvLayer2D, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2c = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        x2l = self.conv1a(X)
        x2l = self.conv1b(x2l)
        x2r = self.conv2a(X)
        x2r = self.conv2b(x2r)
        x2r = self.conv2c(x2r)

        # print(x2l.shape, x2r.shape)
        x10 = torch.add(x2l, x2r)
        # print(x10.shape)
        return self.conv3(x10)


class CMA2D(nn.Module):  # Channel Merge Attention
    def __init__(self, in_channels):
        super(CMA2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1)  # // 4 考虑输出是否要除以4或6，当然如果除的话可以说是减少参数量
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # global average pool
        x1 = self.avg_pool(x)
        x1 = self.conv1_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.sigmoid(x1)  # output N * C ?
        x2 = x * x1
        x2 = x + x2
        x2 = self.conv2(x2)  # 增加的
        return x2


class FGC2D(nn.Module):
    def __init__(self, time_num, band_num, class_num):
        super(FGC2D, self).__init__()
        self.band_num = band_num*time_num
        self.class_num = class_num
        self.name = 'FGC2D'

        channels = [32, 64, 128, 256]

        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        self.down_channel1 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_channel2 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Sequential(
            CMA2D(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMA2D(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            CMA2D(channels[1]),
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv8 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x1 = self.down_channel1(conv4)
        x2 = self.avg_pool(x1)
        x2 = self.down_channel2(x2)
        x2 = self.sigmoid(x2)  # output N * C ?
        x2 = x2 * x1 + x1
        x2 = self.down_channel3(x2)

        deconv3 = self.deconv3(x2)
        conv5 = torch.cat((deconv3, conv3), 1)
        conv5 = self.conv5(conv5)

        deconv2 = self.deconv2(conv5)
        conv6 = torch.cat((deconv2, conv2), 1)
        conv6 = self.conv6(conv6)

        deconv1 = self.deconv1(conv6)
        conv7 = torch.cat((deconv1, conv1), 1)
        conv7 = self.conv7(conv7)

        output = self.conv8(conv7)

        return output


class UNet2D(nn.Module):
    def __init__(self, time_num, band_num, class_num):
        super(UNet2D, self).__init__()
        self.band_num = band_num * time_num
        self.class_num = class_num
        self.name = 'UNet2D'

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


class MSFCN2D(nn.Module):  # 通道融合，全局池化
    def __init__(self, time_num, band_num, class_num):
        super(MSFCN2D, self).__init__()
        self.band_num = time_num * band_num
        self.class_num = class_num
        self.name = 'MSFCN2D'
        channels = [64, 128, 256, 512]

        self.conv1 = nn.Sequential(
            DualConvLayer2D(self.band_num, channels[0])
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[0], channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[1], channels[2])
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[2], channels[3])
        )
        self.down_channel1 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_channel2 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Sequential(
            CMA2D(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMA2D(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            CMA2D(channels[1]),
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv8 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x1 = self.down_channel1(conv4)
        x2 = self.avg_pool(x1)
        x2 = self.down_channel2(x2)
        x2 = self.sigmoid(x2)
        x2 = x2 * x1 + x1
        x2 = self.down_channel3(x2)

        deconv3 = self.deconv3(x2)
        conv5 = torch.cat((deconv3, conv3), 1)
        conv5 = self.conv5(conv5)

        deconv2 = self.deconv2(conv5)
        conv6 = torch.cat((deconv2, conv2), 1)
        conv6 = self.conv6(conv6)

        deconv1 = self.deconv1(conv6)
        conv7 = torch.cat((deconv1, conv1), 1)
        conv7 = self.conv7(conv7)
        output = self.conv8(conv7)
        del conv1, conv2, conv3, conv4, conv5, conv6, conv7, deconv1, deconv2, deconv3, x, x1, x2

        return output


if __name__ == '__main__':
    '''
    FCG2D,
    UNet2D
    '''
    time_num = 4
    band_num = 4
    class_num = 4
    model = UNet2D(time_num, band_num, class_num)
    flops, params = profile(model, input_size=(1, band_num*time_num, 256, 256))
    model.cuda()
    torchsummary.summary(model, (band_num*time_num, 256, 256))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
