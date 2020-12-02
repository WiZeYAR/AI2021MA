import torch
from torchsummary import summary
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, init, Dropout



class CNN_for_GoodEdgeDistribution(torch.nn.Module):

    def __init__(self):
        super(CNN_for_GoodEdgeDistribution, self).__init__()
        self.input_dim = 1


        # layers of Conv
        self.conv1  = Conv2d(in_channels=self.input_dim, out_channels=32,
                             kernel_size=5, stride=1, padding=2)
        self.bn1    = BatchNorm2d(32)

        self.conv2  = Conv2d(in_channels=32, out_channels=32, kernel_size=6, stride=2, padding=2)
        self.conv3  = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3    = BatchNorm2d(64)

        self.conv4  = Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.conv5  = Conv2d(in_channels=64, out_channels=96, kernel_size=7, stride=1, padding=3)
        self.bn5    = BatchNorm2d(96)

        self.conv6  = Conv2d(in_channels=96, out_channels=96, kernel_size=8, stride=2, padding=3)
        self.drop6 = Dropout(p=0.7)

        self.conv7  = Conv2d(in_channels=96, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.drop7 = Dropout(p=0.7)
        self.bn7    = BatchNorm2d(128)

        self.conv8  = Conv2d(in_channels=128, out_channels=128, kernel_size=8, stride=2, padding=3)
        self.drop8 = Dropout(p=0.7)

        self.conv9  = Conv2d(in_channels=128, out_channels=160, kernel_size=7, stride=1, padding=3)
        self.drop9 = Dropout(p=0.7)
        self.bn9    = BatchNorm2d(160 + 128)

        self.conv10 = Conv2d(in_channels=160, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.drop10 = Dropout(p=0.7)
        self.bn10   = BatchNorm2d(128)

        self.conv11 = Conv2d(in_channels=128, out_channels=96, kernel_size=7, stride=1, padding=3)
        self.drop11 = Dropout(p=0.7)
        self.bn11   = BatchNorm2d(96)

        self.conv12 = Conv2d(in_channels=96, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn12   = BatchNorm2d(64)

        self.conv13 = Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn13   = BatchNorm2d(32)

        self.conv14 = Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.bn14   = BatchNorm2d(1)


        # layer of Transposed Conv
        self.tran_conv1 = ConvTranspose2d(in_channels=64*2, out_channels=64,
                                          kernel_size=6, stride=2, padding=2)
        self.tran_conv2 = ConvTranspose2d(in_channels=96*2, out_channels=96,
                                          kernel_size=8, stride=2, padding=3)
        self.tran_conv3 = ConvTranspose2d(in_channels=128*2, out_channels=128,
                                          kernel_size=8, stride=2, padding=3)
        self.drop3_ = Dropout(p=0.7)

        self.tran_conv4 = ConvTranspose2d(in_channels=128 + 160, out_channels=160,
                                          kernel_size=8, stride=2, padding=3)
        self.drop4_ = Dropout(p=0.7)


        # activation functions
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)
        self.identity = torch.nn.Identity()
        self.sigmoid = torch.nn.Sigmoid()

    def weight_init(self, m):
        if isinstance(m, Conv2d):
            init.xavier_normal_(m.weight, gain=init.calculate_gain('relu'))
            init.zeros_(m.bias)
        elif isinstance(m, ConvTranspose2d):
            init.xavier_normal_(m.weight)
            init.zeros_(m.bias)

    def forward(self, X):
        # ENCODER
        out = self.leaky(self.conv1(X))
        out = self.bn1(out)
        self.out_1_cnn = out

        out = self.leaky(self.conv2(out))
        out1 = self.leaky(self.conv3(out))
        out1 = self.bn3(out1)
        self.out_3_cnn = out1

        out = self.leaky(self.conv4(out1))
        out2 = self.leaky(self.conv5(out))
        out2 = self.bn5(out2)
        self.out_5_cnn = out2

        out = self.drop6(self.leaky(self.conv6(out2)))
        out3 = self.drop7(self.leaky(self.conv7(out)))

        out3 = self.bn7(out3)
        self.out_7_cnn = out3

        out4 = self.drop8(self.leaky(self.conv8(out3)))
        self.out_8_cnn = out4

        # DECODER
        out = self.drop9(self.leaky(self.conv9(out4)))

        out = torch.cat([out, out4], 1)
        out = self.bn9(out)
        self.out_9_cnn = out

        out = self.drop4_(self.leaky(self.tran_conv4(out)))
        out = self.drop10(self.leaky(self.conv10(out)))

        out = self.bn10(out)
        self.out_10_cnn = out

        out = torch.cat([out, out3], 1)
        out = self.drop3_(self.leaky(self.tran_conv3(out)))
        out = self.drop11(self.leaky(self.conv11(out)))

        out = self.bn11(out)
        self.out_11_cnn = out

        out = torch.cat([out, out2], 1)
        out = self.leaky(self.tran_conv2(out))
        out = self.leaky(self.conv12(out))
        out = self.bn12(out)
        self.out_12_cnn = out

        out = torch.cat([out, out1], 1)
        out = self.leaky(self.tran_conv1(out))
        out = self.leaky(self.conv13(out))
        out = self.bn13(out)
        self.out_13_cnn = out

        out = self.sigmoid(self.conv14(out))
        return self.identity(out)



def test_net(l):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CNN_for_GoodEdgeDistribution().to(device)
    summary(net, (1, 192, 192))


test_net('test')