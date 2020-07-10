from .DG_Module import *

class Up_cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(out_channels))
        self.ReLU =nn.ReLU()

    def forward(self, x1, x2):
        x2 = self.up(x2)
        w_out = x1.shape[-1]
        h_out = x1.shape[-2]
        x2 = F.interpolate(x2, size=[h_out, w_out], mode='bilinear')

        return self.ReLU(x1 + x2)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  Residual_Block(in_channels, out_channels))

    def forward(self, x):
        return self.down(x)


class PDR_UPPNet(nn.Module):
    def __init__(self, n_ch, n_classes, ocr=False, check=True):
        super(PDR_UPPNet, self).__init__()
        self.n_classes= n_classes
        self.check = check
        channels = [64, 128, 256, 512, 1024]
        self.input_layer = Residual_Block(n_ch, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        self.Up0_1 = Up_cat(channels[1], channels[0])
        self.Conv0_1 = Residual_Block(channels[0], channels[0])

        self.Up1_1 = Up_cat(channels[2], channels[1])
        self.Conv1_1 = Residual_Block(channels[1], channels[1])

        self.Up0_2 = Up_cat(channels[1], channels[0])
        self.Conv0_2 = Residual_Block(channels[0]*2, channels[0])

        self.Up2_1 = Up_cat(channels[3], channels[2])
        self.Conv2_1 = Residual_Block(channels[2], channels[2])

        self.Up1_2 = Up_cat(channels[2], channels[1])
        self.Conv1_2 = Residual_Block(channels[1]*2, channels[1])

        self.Up0_3 = Up_cat(channels[1], channels[0])
        self.Conv0_3 = Residual_Block(channels[0]*3, channels[0])

        self.Up3_1 = Up_cat(channels[4], channels[3])
        self.Conv3_1 = Residual_Block(channels[3], channels[3])

        self.Up2_2 = Up_cat(channels[3], channels[2])
        self.Conv2_2 = Residual_Block(channels[2]*2, channels[2])

        self.Up1_3 = Up_cat(channels[2], channels[1])
        self.Conv1_3 = Residual_Block(channels[1]*3, channels[1])

        self.Up0_4 = Up_cat(channels[1], channels[0])
        self.Conv0_4 = Residual_Block(channels[0]*4, channels[0])

        if ocr is True:
            self.out = OCR(channels[0], n_classes)
            print('OCR')
        else:
            self.out = Out(channels[0], n_classes)
        if self.check is True:
            self.last_layer1 = nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1, padding=0)
            self.last_layer2 = nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1, padding=0)
            self.last_layer3 = nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1, padding=0)
            print('check')

    def forward(self, x):
        x = self.input_layer(x)
        x1_0 = self.down1(x)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.Up0_1(x, x1_0)
        x0_1 = self.Conv0_1(x0_1)
        x1_1 = self.Up1_1(x1_0, x2_0)
        x1_1 = self.Conv1_1(x1_1)
        x2_1 = self.Up2_1(x2_0, x3_0)
        x2_1 = self.Conv2_1(x2_1)
        x3_1 = self.Up3_1(x3_0, x4_0)
        x3_1 = self.Conv3_1(x3_1)

        x0_2 = self.Up0_2(x0_1, x1_1)
        x0_2 = self.Conv0_2(torch.cat([x0_2, x], dim=1))
        x1_2 = self.Up1_2(x1_1, x2_1)
        x1_2 = self.Conv1_2(torch.cat([x1_2, x1_0], dim=1))
        x2_2 = self.Up2_2(x2_1, x3_1)
        x2_2 = self.Conv2_2(torch.cat([x2_2, x2_0], dim=1))

        x0_3 = self.Up0_3(x0_2, x1_2)
        x0_3 = self.Conv0_3(torch.cat([x0_3, x0_1, x], dim=1))
        x1_3 = self.Up1_3(x1_2, x2_2)
        x1_3 = self.Conv1_3(torch.cat([x1_3, x1_1, x1_0], dim=1))

        x0_4 = self.Up0_4(x0_3, x1_3)
        x0_4 = self.Conv0_4(torch.cat([x0_4, x0_2, x0_1, x], dim=1))

        out_p = self.out(x0_4)

        if self.check is True:
            check1 = self.last_layer1(x0_3)
            check2 = self.last_layer1(x0_2)
            check3 = self.last_layer1(x0_1)
            if not isinstance(out_p, list):
                out = []
                out.append(out_p)
            else:
                out = out_p
            out.append(check1)
            out.append(check2)
            out.append(check3)
        else:
            out = out_p

        return out