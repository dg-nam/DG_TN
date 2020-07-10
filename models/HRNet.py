from .DG_Module import *
import torch._utils
import numpy as np


#총 4개의 stage로 구분할 때 각 stage module
class HR_Module(nn.Module):
    def __init__(self, num_branches, num_blocks, num_channels,  multi_scale_output=True):
        super(HR_Module, self).__init__()
        self.num_channels = num_channels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self.make_branches(num_branches, num_blocks, num_channels)

        self.fuse_layers = self.make_fuse_layers(num_branches, num_channels)
        self.ReLU = nn.ReLU()

    # branch하나에서 ConvBlock을 block수 만큼 반복
    def make_one_branch(self, branch_index, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(Residual_Block(num_channels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    # 모든 branch에서 적용
    def make_branches(self, num_branches, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self.make_one_branch(i, num_blocks, num_channels))

        return nn.ModuleList(branches)

    # 각 branch의 결과들을 fully connected로 연결하는 Layer Module
    def make_fuse_layers(self, num_branches, num_channels):

        fuse_layers = []
        #i는 Output 부분 j는 input 부분의 순서(고화질부터)
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_channels[j], num_channels[i],
                                                              kernel_size=1,
                                                              stride=1,
                                                              padding=0,
                                                              bias=False),
                                                    nn.BatchNorm2d(num_channels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    down3x3 = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            down3x3.append(nn.Sequential(
                                nn.Conv2d(num_channels[j + k], num_channels[j + k + 1],
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          bias=False),
                                nn.BatchNorm2d(num_channels[j + k + 1])))
                        else:
                            down3x3.append(nn.Sequential(
                                nn.Conv2d(num_channels[j + k], num_channels[j + k + 1],
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          bias=False),
                                nn.BatchNorm2d(num_channels[j + k + 1]),
                                nn.ReLU()))
                    fuse_layer.append(nn.Sequential(*down3x3))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        # branch별로 convblock 반복
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # branch 결과들 정보교환
        x_fuse = []
        #j input, i output
        for i in range(self.num_branches):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    w_output = x[i].shape[-1]
                    h_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[h_output, w_output], mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.ReLU(y))

        return x_fuse


# main Network Model
class HRNet(nn.Module):
    def __init__(self, n_ch, n_classes, ocr=False):
        super(HRNet, self).__init__()

        self.n_classes = n_classes
        # start (해상도 1/4로 낮추는 과정)
        self.conv1 = nn.Conv2d(n_ch, 64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.BN1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.BN2 = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()

        stage1_num_branches = 1
        stage1_num_blocks = [4]
        stage1_num_channels = [64]
        self.layer1 = self.make_layer(64, stage1_num_channels[0], stage1_num_blocks[0])

        self.stage2_num_branches = 2
        stage2_num_blocks = [4, 4]
        stage2_num_channels = [64, 128]
        self.transition1 = self.make_transition_layer(stage1_num_channels, stage2_num_channels)
        self.stage2 = HR_Module(self.stage2_num_branches, stage2_num_blocks,
                                      stage2_num_channels, multi_scale_output=True)

        self.stage3_num_branches = 3
        stage3_num_blocks = [4, 4, 4]
        stage3_num_channels = [64, 128, 256]
        self.transition2 = self.make_transition_layer(stage2_num_channels, stage3_num_channels)
        self.stage3 = HR_Module(self.stage3_num_branches, stage3_num_blocks,
                                      stage3_num_channels, multi_scale_output=True)

        self.stage4_num_branches = 4
        stage4_num_blocks = [4, 4, 4, 4]
        stage4_num_channels = [64, 128, 256, 512]
        self.transition3 = self.make_transition_layer(stage3_num_channels, stage4_num_channels)
        self.stage4 = HR_Module(self.stage4_num_branches, stage4_num_blocks,
                                      stage4_num_channels, multi_scale_output=True)

        last_input_channels = np.int(np.sum(stage4_num_channels))

        if ocr is True:
            self.out = OCR(last_input_channels, n_classes)
        else:
            self.out = Out(last_input_channels, n_classes)

    # branch fully connected 이후 새로운 branch 생성
    def make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                transition_layers.append(None)

            else:
                conv3x3 = []
                in_channels = num_channels_pre_layer[-1]
                out_channels = num_channels_cur_layer[i]
                conv3x3.append(nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                        kernel_size=3,
                                                        stride=2,
                                                        padding=1,
                                                        bias=False),
                                              nn.BatchNorm2d(out_channels),
                                              nn.ReLU()))
                transition_layers.append(*conv3x3)

        return nn.ModuleList(transition_layers)

    #branch 1개일 때 Convblock 반복
    def make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(Residual_Block(in_channels, out_channels))
        for i in range(1, blocks):
            layers.append(Residual_Block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        in_x_h, in_x_w = x.size(2), x.size(3)
        #input resolution 1/2로 낮춤
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.ReLU(x)
        #stage 1
        x = self.layer1(x)

        #2nd branch 생성
        x_list = []
        for i in range(self.stage2_num_branches):
            if i == self.stage2_num_branches -1:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        #stage 2
        y_list = self.stage2(x_list)

        #3rd branch 생성
        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        #stage 3
        y_list = self.stage3(x_list)

        #4th branch 생성
        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        #stage 4
        x = self.stage4(x_list)

        #Output Layer
        x_h, x_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x_h, x_w), mode='bilinear')
        x2 = F.interpolate(x[2], size=(x_h, x_w), mode='bilinear')
        x3 = F.interpolate(x[3], size=(x_h, x_w), mode='bilinear')

        feats = torch.cat([x[0], x1, x2, x3], 1)

        out_seg = self.out(feats)

        if (isinstance(out_seg, tuple) or isinstance(out_seg, list)) and len(out_seg) == 2:
            out_seg[0] = F.interpolate(out_seg[0], size=(in_x_h, in_x_w), mode='bilinear')
            out_seg[1] = F.interpolate(out_seg[1], size=(in_x_h, in_x_w), mode='bilinear')
        else:
            out_seg = F.interpolate(out_seg, size=(in_x_h, in_x_w), mode='bilinear')

        return out_seg