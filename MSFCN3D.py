from torch import nn
import torch


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    if kernel_size is None:
        kernel_size = 3

    if stride is None:
        stride = 1

    if padding is None:
        padding = 1

    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)
    )


class DualConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DualConvLayer, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2c = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        x2l = self.conv1a(X)
        x2l = self.conv1b(x2l)
        x2r = self.conv2a(X)
        x2r = self.conv2b(x2r)
        x2r = self.conv2c(x2r)

        x10 = torch.add(x2l, x2r)
        return self.conv3(x10)


class CMA(nn.Module):
    def __init__(self, in_channels):
        super(CMA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv1_1 = nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = self.conv1_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.sigmoid(x1)
        x2 = x * x1
        x2 = x + x2
        x2 = self.conv2(x2)
        return x2


class MSFCN(nn.Module):
    def __init__(self, time_num, band_num, class_num):
        super(MSFCN, self).__init__()
        self.time_num = time_num
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'MSFCN'
        channels = [32, 64, 128, 256]

        self.conv1 = DualConvLayer(band_num, channels[0])

        self.conv2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            DualConvLayer(channels[0], channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            DualConvLayer(channels[1], channels[2])
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            DualConvLayer(channels[2], channels[3])
        )
        self.down_channel1 = nn.Conv3d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.down_channel2 = nn.Conv3d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv3d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv5 = nn.Sequential(
            CMA(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv6 = nn.Sequential(
            CMA(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv7 = nn.Sequential(
            CMA(channels[1]),
            conv3otherRelu(channels[1], channels[0], kernel_size=(self.time_num, 3, 3), stride=(self.time_num, 1, 1),
                           padding=(0, 1, 1)),
        )

        self.conv8 = nn.Conv3d(channels[0], self.class_num, kernel_size=1, stride=1)

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

        return output.squeeze(-3)


if __name__ == '__main__':
    batch_size = 16
    height = 64
    weight = 64
    time_num = 4
    band_num = 4
    class_num = 4
    x = torch.randn(batch_size, time_num, band_num, height, weight).cuda()
    model = MSFCN(time_num, band_num, class_num).cuda()
    y = model(x)


