import torch
import torch.nn as nn
import torch.nn.functional as F

class Double33Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double33conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(out_ch),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(out_ch),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double33conv(x)

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_Block, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        if in_channels != out_channels:
            self.tran_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            residual = x
        else:
            x = self.tran_channels(x)
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual
        out = self.ReLU(x)

        return out

class Dilated_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch_1 = int(out_ch/2)
        mid_ch_2 = int(out_ch/4)
        mid_ch_3 = int(out_ch/8)
        mid_ch_4 = int(out_ch/16)

        self.conv33 = nn.Sequential(nn.Conv2d(in_ch, mid_ch_1, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(mid_ch_1),
                                    nn.ReLU())
        self.dil_conv1 = nn.Sequential(nn.Conv2d(mid_ch_1, mid_ch_2, kernel_size=3, padding=2, dilation=2),
                                       nn.BatchNorm2d(mid_ch_2),
                                       nn.ReLU())
        self.dil_conv2 = nn.Sequential(nn.Conv2d(mid_ch_2, mid_ch_3, kernel_size=3, padding=2, dilation=2),
                                       nn.BatchNorm2d(mid_ch_3),
                                       nn.ReLU())
        self.dil_conv3 = nn.Sequential(nn.Conv2d(mid_ch_3, mid_ch_4, kernel_size=3, padding=2, dilation=2),
                                       nn.BatchNorm2d(mid_ch_4),
                                       nn.ReLU())
        self.dil_conv4 = nn.Sequential(nn.Conv2d(mid_ch_4, mid_ch_4, kernel_size=3, padding=2, dilation=2),
                                       nn.BatchNorm2d(mid_ch_4),
                                       nn.ReLU())

    def forward(self, x):
        x = self.conv33(x)
        x1 = self.dil_conv1(x)
        x2 = self.dil_conv2(x1)
        x3 = self.dil_conv3(x2)
        x4 = self.dil_conv4(x3)

        return torch.cat([x, x1, x2, x3, x4], dim=1)

class P_Dilated_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch_1 = int(out_ch/2)
        mid_ch_2 = int(out_ch/4)
        mid_ch_3 = int(out_ch/8)
        mid_ch_4 = int(out_ch/16)

        self.conv33 = nn.Sequential(nn.Conv2d(in_ch, mid_ch_1, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(mid_ch_1),
                                    nn.ReLU())
        self.dil_conv1 = nn.Sequential(nn.Conv2d(in_ch, mid_ch_2, kernel_size=3, padding=2, dilation=2),
                                       nn.BatchNorm2d(mid_ch_2),
                                       nn.ReLU())
        self.dil_conv2 = nn.Sequential(nn.Conv2d(in_ch, mid_ch_3, kernel_size=3, padding=3, dilation=3),
                                       nn.BatchNorm2d(mid_ch_3),
                                       nn.ReLU())
        self.dil_conv3 = nn.Sequential(nn.Conv2d(in_ch, mid_ch_4, kernel_size=3, padding=4, dilation=4),
                                       nn.BatchNorm2d(mid_ch_4),
                                       nn.ReLU())
        self.dil_conv4 = nn.Sequential(nn.Conv2d(in_ch, mid_ch_4, kernel_size=3, padding=5, dilation=5),
                                       nn.BatchNorm2d(mid_ch_4),
                                       nn.ReLU())

    def forward(self, x):
        x1 = self.conv33(x)
        x2 = self.dil_conv1(x)
        x3 = self.dil_conv2(x)
        x4 = self.dil_conv3(x)
        x5 = self.dil_conv4(x)

        return torch.cat([x1, x2, x3, x4, x5], dim=1)

class OCR(nn.Module):
    def __init__(self, last_input_channel, n_classes):
        super(OCR, self).__init__()
        self.n_classes = n_classes
        self.last_input_channel = last_input_channel
        self.out = Out(self.last_input_channel, self.n_classes)
        self.ocr_gather_head = SpatialGather_Module(self.n_classes)

        ocr_mid_channels = int(self.last_input_channel / 2)
        ocr_key_channels = int(ocr_mid_channels / 2)
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.last_input_channel, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU()
        )

        self.ocr_distri_head = SpatialOCR_Module(ocr_mid_channels,
                                                 ocr_key_channels,
                                                 ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05)

        self.cls_head = nn.Conv2d(ocr_mid_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(self.last_input_channel, self.last_input_channel,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.last_input_channel),
            nn.ReLU(),
            nn.Conv2d(self.last_input_channel, n_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, feats):
        out_aux_seg = []

        out_aux = self.out(feats)

        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)

        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out)
        out_aux_seg.append(out_aux)

        return out_aux_seg

class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, channels, height, width = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, channels, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # bat x hw x c
        probs = F.softmax(self.scale * probs, dim=2) # bat x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3) # bat x k x c
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(self.key_channels),
            #nn.ReLU(),
            #nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(self.key_channels),
            #nn.ReLU()
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(self.key_channels),
            #nn.ReLU()
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(self.key_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim =1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)

        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context

class ObjectAttentionBlock2D(ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale = 1):
        super(ObjectAttentionBlock2D, self).__init__(in_channels, key_channels, scale)

class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.out(x)

class Fully_connect(nn.Module):
    def __init__(self, n_classes, H, W):
        super().__init__()
        self.Full = nn.Sequential(
            nn.Linear(64 * H * W, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes + 1)
        )

    def forward(self, x):
        out = self.Full(x)

        return out
