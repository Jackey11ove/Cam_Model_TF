import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/workspace/I/wangyuanzhe0/Cam_Model_TF')

from module import *
from utils.general import check_anchor_order


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g)

    def forward(self,x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.conv.quantize(quant_type, num_bits, e_bits)
        self.qcat = QConcat(quant_type, 4, num_bits=num_bits, e_bits=e_bits)

    def quantize_forward(self,x):
        qarray = [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]]
        cat_out = self.qcat(qarray)
        out = self.conv.quantize_forward(cat_out)
        return out
    
    def freeze(self):
        self.qcat.freeze()
        self.conv.freeze()

    def fakefreeze(self):
        self.conv.fakefreeze()

    def quantize_inference(self,x):
        qarray = [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]]
        cat_out = self.qcat.quantize_inference(qarray)
        out = self.conv.quantize_inference(cat_out)
        return out

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, 0.001, 0.03)
        self.lkrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self,x):
        return self.lkrelu(self.bn(self.conv(x)))
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.qconv = QConv2d(quant_type, self.conv, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qbn = QBN(quant_type, self.bn, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qlkrelu = QLeakyReLU(0.1, quant_type, qi=True, qo=False, num_bits=num_bits, e_bits=e_bits)

    def quantize_forward(self, x):
        x = self.qconv(x)
        x = self.qbn(x)
        x = self.qlkrelu(x)
        return x
    
    def freeze(self):
        self.qconv.freeze()
        self.qbn.freeze()
        self.qlkrelu.freeze()

    def fakefreeze(self):
        self.qconv.fakefreeze()
        self.qbn.fakefreeze()
        self.qlkrelu.fakefreeze()

    def quantize_inference(self, x):
        qx = self.qconv.quantize_inference(x)
        qx = self.qbn.quantize_inference(qx)
        qx = self.qlkrelu.quantize_inference(qx)
        return qx
    
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.cv1.quantize(quant_type, num_bits, e_bits)
        self.cv2.quantize(quant_type, num_bits, e_bits)
        self.qadd = QElementwiseAdd(quant_type, 2, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        if self.add:
            qx = self.cv1.quantize_forward(x)
            qx = self.cv2.quantize_forward(qx)
            qarray = [x,qx]
            out = self.qadd(qarray)
            return out
        else:
            x = self.cv1.quantize_forward(x)
            x = self.cv2.quantize_forward(x)
            return x
        
    def freeze(self):
        if self.add:
            self.cv1.freeze()
            self.cv2.freeze()
            self.qadd.freeze()
        else:
            self.cv1.freeze()
            self.cv2.freeze()

    def fakefreeze(self):
        self.cv1.fakefreeze()
        self.cv2.fakefreeze()

    def quantize_inference(self,x):
        if self.add:
            qx = self.cv1.quantize_inference(x)
            qx = self.cv2.quantize_inference(qx)
            qarray = [x,qx]
            out = self.qadd.quantize_inference(qarray)
            return out
        else:
            qx = self.cv1.quantize_inference(x)
            qx = self.cv2.quantize_inference(qx)
            return qx
        
class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_, 0.001, 0.03)  # applied to cat(cv2, cv3)
        self.lkrelu = nn.LeakyReLU(0.1, inplace=True)
        self.Bottleneck_layers = nn.ModuleList([Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    
    def forward(self,x):
        y1 = self.cv1(x)
        for Bottleneck in self.Bottleneck_layers:
            y1 = Bottleneck(y1)
        y1 = self.cv3(y1)
        y2 = self.cv2(x)
        out = self.cv4(self.lkrelu(self.bn(torch.cat((y1,y2), dim=1))))
        return out
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.cv1.quantize(quant_type, num_bits, e_bits)
        self.qcv2 = QConv2d(quant_type, self.cv2, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qcv3 = QConv2d(quant_type, self.cv3, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.cv4.quantize(quant_type, num_bits, e_bits)
        self.qbn = QBN(quant_type, self.bn, num_bits=num_bits, e_bits=e_bits)
        self.qlkrelu = QLeakyReLU(0.1, quant_type, qi=True, qo=False, num_bits=num_bits, e_bits=e_bits)
        for Bottleneck in self.Bottleneck_layers:
            Bottleneck.quantize(quant_type, num_bits, e_bits)
        self.qcat = QConcat(quant_type, 2, num_bits=num_bits, e_bits=e_bits)

    def quantize_forward(self,x):
        y1 = self.cv1.quantize_forward(x)
        for Bottleneck in self.Bottleneck_layers:
            y1 = Bottleneck.quantize_forward(y1)
        y1 = self.qcv3(y1)
        y2 = self.qcv2(x)
        qarray = [y1,y2]
        cat_out = self.qcat(qarray)
        bn_out = self.qbn(cat_out)
        lkrelu_out = self.qlkrelu(bn_out)
        out = self.cv4.quantize_forward(lkrelu_out)
        return out
    
    def freeze(self):
        self.cv1.freeze()
        for Bottleneck in self.Bottleneck_layers:
            Bottleneck.freeze()
        self.qcv3.freeze()
        self.qcv2.freeze()
        self.qcat.freeze()
        self.qbn.freeze()
        self.qlkrelu.freeze()
        self.cv4.freeze()
    
    def fakefreeze(self):
        self.cv1.fakefreeze()
        for Bottleneck in self.Bottleneck_layers:
            Bottleneck.fakefreeze()
        self.qcv3.fakefreeze()
        self.qcv2.fakefreeze()
        self.qbn.fakefreeze()
        self.qlkrelu.fakefreeze()
        self.cv4.fakefreeze()

    def quantize_inference(self,x):
        y1 = self.cv1.quantize_inference(x)
        for Bottleneck in self.Bottleneck_layers:
            y1 = Bottleneck.quantize_inference(y1)
        y1 = self.qcv3.quantize_inference(y1)
        y2 = self.qcv2.quantize_inference(x)
        qarray = [y1,y2]
        cat_out = self.qcat.quantize_inference(qarray)
        bn_out = self.qbn.quantize_inference(cat_out)
        lkrelu_out = self.qlkrelu.quantize_inference(bn_out)
        out = self.cv4.quantize_inference(lkrelu_out)
        return out
    
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.k = k
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    
    def forward(self,x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.cv1.quantize(quant_type, num_bits, e_bits)
        self.cv2.quantize(quant_type, num_bits, e_bits)
        self.qmaxpool_layers = nn.ModuleList([QMaxPooling2d(quant_type, x, 1, x//2, num_bits=num_bits, e_bits=e_bits) for x in self.k])
        self.qcat = QConcat(quant_type, 4, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self,x):
        x = self.cv1.quantize_forward(x)
        qarray = [x]
        for layer in self.qmaxpool_layers:
            maxpool_out = layer(x)
            qarray.append(maxpool_out)
        cat_out = self.qcat(qarray)
        out = self.cv2.quantize_forward(cat_out)
        return out
    
    def freeze(self):
        self.cv1.freeze()
        for layer in self.qmaxpool_layers:
            layer.freeze()
        self.qcat.freeze()
        self.cv2.freeze()

    def fakefreeze(self):
        self.cv1.fakefreeze()
        for layer in self.qmaxpool_layers:
            layer.fakefreeze()
        self.cv2.fakefreeze()

    def quantize_inference(self,x):
        qx = self.cv1.quantize_inference(x)
        qarray = [qx]
        for layer in self.qmaxpool_layers:
            maxpool_out = layer.quantize_inference(qx)
            qarray.append(maxpool_out)
        cat_out = self.qcat.quantize_inference(qarray)
        out = self.cv2.quantize_inference(cat_out)
        return out

class ResBlock(nn.Module):
    # SPP_flag = True means this layer has SPP
    # Conv & BottleneckCSP has the same c_out, n is the param for BottleneckCSP
    def __init__(self, SPP_flag, c_in, c_out, n):
        super(ResBlock, self).__init__()
        self.SPP_flag = SPP_flag
        self.c_in = c_in
        self.c_out = c_out
        self.n = n
        self.Conv = Conv(self.c_in, self.c_out, 3, 2)
        if self.SPP_flag:
            self.SPP = SPP(self.c_out, self.c_out)
        shortcut = not self.SPP_flag
        self.BottleneckCSP = BottleneckCSP(self.c_out, self.c_out, n, shortcut)
    
    def forward(self,x):
        x = self.Conv(x)
        if self.SPP_flag == True:
            x = self.SPP(x)
        x = self.BottleneckCSP(x)
        return x
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.Conv.quantize(quant_type, num_bits, e_bits)
        if self.SPP_flag == True:
            self.SPP.quantize(quant_type, num_bits, e_bits)
        self.BottleneckCSP.quantize(quant_type, num_bits, e_bits)

    def quantize_forward(self,x):
        x = self.Conv.quantize_forward(x)
        if self.SPP_flag == True:
            x = self.SPP.quantize_forward(x)
        x = self.BottleneckCSP.quantize_forward(x)
        return x
    
    def freeze(self):
        self.Conv.freeze()
        if self.SPP_flag == True:
            self.SPP.freeze()
        self.BottleneckCSP.freeze()

    def fakefreeze(self):
        self.Conv.fakefreeze()
        if self.SPP_flag == True:
            self.SPP.fakefreeze()
        self.BottleneckCSP.fakefreeze()

    def quantize_inference(self,x):
        qx = self.Conv.quantize_inference(x)
        if self.SPP_flag == True:
            qx = self.SPP.quantize_inference(qx)
        qx = self.BottleneckCSP.quantize_inference(qx)
        return qx
    
class CSPDarknet(nn.Module):
    def __init__(self, c_in):
        super(CSPDarknet, self).__init__()
        self.c_in = c_in
        self.Focus = Focus(c_in, 32, 3)
        self.ResBlock1 = ResBlock(False, 32, 64, 1)
        self.ResBlock2 = ResBlock(False, 64, 128, 3)
        self.ResBlock3 = ResBlock(False, 128, 256, 3)
        self.ResBlock4 = ResBlock(True, 256, 512, 1)

    def forward(self,x):
        x = self.Focus(x)
        Block1_out = self.ResBlock1(x)
        Block2_out = self.ResBlock2(Block1_out)
        Block3_out = self.ResBlock3(Block2_out)
        Block4_out = self.ResBlock4(Block3_out)
        return Block2_out, Block3_out, Block4_out
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.Focus.quantize(quant_type, num_bits, e_bits)
        self.ResBlock1.quantize(quant_type, num_bits, e_bits)
        self.ResBlock2.quantize(quant_type, num_bits, e_bits)
        self.ResBlock3.quantize(quant_type, num_bits, e_bits)
        self.ResBlock4.quantize(quant_type, num_bits, e_bits)

    def quantize_forward(self,x):
        x = self.Focus.quantize_forward(x)
        Block1_out = self.ResBlock1.quantize_forward(x)
        Block2_out = self.ResBlock2.quantize_forward(Block1_out)
        Block3_out = self.ResBlock3.quantize_forward(Block2_out)
        Block4_out = self.ResBlock4.quantize_forward(Block3_out)
        return Block2_out, Block3_out, Block4_out
    
    def freeze(self):
        self.Focus.freeze()
        self.ResBlock1.freeze()
        self.ResBlock2.freeze()
        self.ResBlock3.freeze()
        self.ResBlock4.freeze()
    
    def fakefreeze(self):
        self.Focus.fakefreeze()
        self.ResBlock1.fakefreeze()
        self.ResBlock2.fakefreeze()
        self.ResBlock3.fakefreeze()
        self.ResBlock4.fakefreeze()

    def quantize_inference(self,x):
        qx = self.Focus.quantize_inference(x)
        Block1_out = self.ResBlock1.quantize_inference(qx)
        Block2_out = self.ResBlock2.quantize_inference(Block1_out)
        Block3_out = self.ResBlock3.quantize_inference(Block2_out)
        Block4_out = self.ResBlock4.quantize_inference(Block3_out)
        return Block2_out, Block3_out, Block4_out

class main_net(nn.Module):
    def __init__(self):
        super(main_net, self).__init__()
        self.CSPDarknet = CSPDarknet(3)

        self.Conv1 = Conv(512, 256, 1, 1)
        self.BottleneckCSP1 = BottleneckCSP(512, 256, 1, False)
        self.Conv2 = Conv(256, 128, 1, 1)
        self.BottleneckCSP2 = BottleneckCSP(256, 128, 1, False)
        self.Conv3 = Conv(128, 128, 3, 2)
        self.BottleneckCSP3 = BottleneckCSP(256, 256, 1, False)
        self.Conv4 = Conv(256, 256, 3, 2)
        self.BottleneckCSP4 = BottleneckCSP(512, 512, 1, False)

        self.Upsample = nn.Upsample(None, 2, 'nearest')

    def forward(self,x):
        feature1, feature2, feature3 = self.CSPDarknet(x)

        Conv1_out = self.Conv1(feature3)
        Upsample1_out = self.Upsample(Conv1_out)
        cat1_out = torch.cat([Upsample1_out, feature2], 1)
        CSP1_out = self.BottleneckCSP1(cat1_out)

        Conv2_out = self.Conv2(CSP1_out)
        Upsample2_out = self.Upsample(Conv2_out)
        cat2_out = torch.cat([Upsample2_out, feature1], 1)
        CSP2_out = self.BottleneckCSP2(cat2_out)
        Head_1 = CSP2_out

        Conv3_out = self.Conv3(CSP2_out)
        cat3_out = torch.cat([Conv3_out, Conv2_out], 1)
        CSP3_out = self.BottleneckCSP3(cat3_out)
        Head_2 = CSP3_out

        Conv4_out = self.Conv4(CSP3_out)
        cat4_out = torch.cat([Conv4_out, Conv1_out], 1)
        CSP4_out = self.BottleneckCSP4(cat4_out)
        Head_3 = CSP4_out

        return Head_1, Head_2, Head_3
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.CSPDarknet.quantize(quant_type, num_bits, e_bits)

        self.Conv1.quantize(quant_type, num_bits, e_bits)
        self.Conv2.quantize(quant_type, num_bits, e_bits)
        self.Conv3.quantize(quant_type, num_bits, e_bits)
        self.Conv4.quantize(quant_type, num_bits, e_bits)

        self.qcat1 = QConcat(quant_type, 2, num_bits=num_bits, e_bits=e_bits)
        self.qcat2 = QConcat(quant_type, 2, num_bits=num_bits, e_bits=e_bits)
        self.qcat3 = QConcat(quant_type, 2, num_bits=num_bits, e_bits=e_bits)
        self.qcat4 = QConcat(quant_type, 2, num_bits=num_bits, e_bits=e_bits)
        
        self.BottleneckCSP1.quantize(quant_type, num_bits, e_bits)
        self.BottleneckCSP2.quantize(quant_type, num_bits, e_bits)
        self.BottleneckCSP3.quantize(quant_type, num_bits, e_bits)
        self.BottleneckCSP4.quantize(quant_type, num_bits, e_bits)

    def quantize_forward(self,x):
        feature1, feature2, feature3 = self.CSPDarknet.quantize_forward(x)

        Conv1_out = self.Conv1.quantize_forward(feature3)
        Upsample1_out = self.Upsample(Conv1_out)
        cat1_out = self.qcat1([Upsample1_out, feature2])
        CSP1_out = self.BottleneckCSP1.quantize_forward(cat1_out)

        Conv2_out = self.Conv2.quantize_forward(CSP1_out)
        Upsample2_out = self.Upsample(Conv2_out)
        cat2_out = self.qcat2([Upsample2_out, feature1])
        CSP2_out = self.BottleneckCSP2.quantize_forward(cat2_out)
        Head_1 = CSP2_out

        Conv3_out = self.Conv3.quantize_forward(CSP2_out)
        cat3_out = self.qcat3([Conv3_out, Conv2_out])
        CSP3_out = self.BottleneckCSP3.quantize_forward(cat3_out)
        Head_2 = CSP3_out

        Conv4_out = self.Conv4.quantize_forward(CSP3_out)
        cat4_out = self.qcat4([Conv4_out, Conv1_out])
        CSP4_out = self.BottleneckCSP4.quantize_forward(cat4_out)
        Head_3 = CSP4_out

        return Head_1, Head_2, Head_3

    def freeze(self):
        self.CSPDarknet.freeze()

        self.Conv1.freeze()
        self.qcat1.freeze()
        self.BottleneckCSP1.freeze()

        self.Conv2.freeze()
        self.qcat2.freeze()
        self.BottleneckCSP2.freeze()

        self.Conv3.freeze()
        self.qcat3.freeze()
        self.BottleneckCSP3.freeze()

        self.Conv4.freeze()
        self.qcat4.freeze()
        self.BottleneckCSP4.freeze()
    
    def fakefreeze(self):
        self.CSPDarknet.fakefreeze()

        self.Conv1.fakefreeze()
        self.BottleneckCSP1.fakefreeze()

        self.Conv2.fakefreeze()
        self.BottleneckCSP2.fakefreeze()

        self.Conv3.fakefreeze()
        self.BottleneckCSP3.fakefreeze()

        self.Conv4.fakefreeze()
        self.BottleneckCSP4.fakefreeze()

    def quantize_inference(self,x):
        feature1, feature2, feature3 = self.CSPDarknet.quantize_inference(x)

        Conv1_out = self.Conv1.quantize_inference(feature3)
        Upsample1_out = self.Upsample(Conv1_out)
        cat1_out = self.qcat1.quantize_inference([Upsample1_out, feature2])
        CSP1_out = self.BottleneckCSP1.quantize_inference(cat1_out)

        Conv2_out = self.Conv2.quantize_inference(CSP1_out)
        Upsample2_out = self.Upsample(Conv2_out)
        cat2_out = self.qcat2.quantize_inference([Upsample2_out, feature1])
        CSP2_out = self.BottleneckCSP2.quantize_inference(cat2_out)
        Head_1 = CSP2_out

        Conv3_out = self.Conv3.quantize_inference(CSP2_out)
        cat3_out = self.qcat3.quantize_inference([Conv3_out, Conv2_out])
        CSP3_out = self.BottleneckCSP3.quantize_inference(cat3_out)
        Head_2 = CSP3_out

        Conv4_out = self.Conv4.quantize_inference(CSP3_out)
        cat4_out = self.qcat4([Conv4_out, Conv1_out])
        CSP4_out = self.BottleneckCSP4.quantize_inference(cat4_out)
        Head_3 = CSP4_out

        return Head_1, Head_2, Head_3
    

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor, 20 + 4 + 1: 种类，框的中心与宽高，置信度
        self.nl = len(anchors)  # number of detection layers, 表示有3个feature要处理
        self.na = len(anchors[0]) // 2  # number of anchors, 表示每个feature中心点对应三个先验框
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  #锚点张量
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        # self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


anchors = [
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326]  # P5/32
]

ch = [128, 256, 512]


class Yolo_v5(nn.Module):
    def __init__(self):
        super(Yolo_v5, self).__init__()
        self.main_net = main_net()
        self.detect_layer = Detect(20, anchors, ch)

        m = self.detect_layer  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect_layer  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():
              b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
              b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self,x):
        Head1, Head2, Head3 = self.main_net(x)
        feature_array = [Head1, Head2, Head3]
        out = self.detect_layer(feature_array)
        return out
    
    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.main_net.quantize(quant_type, num_bits, e_bits)

    def quantize_forward(self,x):
        Head1, Head2, Head3 = self.main_net.quantize_forward(x)
        return Head1, Head2, Head3
    
    def freeze(self):
        self.main_net.freeze()

    def fakefreeze(self):
        self.main_net.fakefreeze()

    def quantize_inference(self,x):
        Head1, Head2, Head3 = self.main_net.quantize_inference(x)
        feature_array = [Head1, Head2, Head3]
        out = self.detect_layer(feature_array)
        return out