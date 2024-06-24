import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from layers.functions.prior_box import PriorBoxFunction
from layers.functions.detection import DetectModule

from data import voc, coco

import sys
sys.path.append(r'C:\Users\Jackey\Desktop\Study\Cambricon_Internship\Model_Transfer\myprj')

from l2norm import L2Norm
from module import *

# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# Vgg_upto_Conv4_3是feature1的路径，截止到vgg的Conv4_3
class Vgg_upto_Conv4_3(nn.Module):

    def __init__(self):
        super(Vgg_upto_Conv4_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1) #conv4-3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2, 2, ceil_mode=True)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))  #conv4-3
        return x

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv1 and relu1
        self.qconv1 = QConv2d(quant_type, self.conv1, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu1 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv2 and relu2
        self.qconv2 = QConv2d(quant_type, self.conv2, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu2 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize first max pool
        self.qmax_pool2d1 = QMaxPooling2d(quant_type, 2, 2, ceil_mode=False, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv3 and relu3
        self.qconv3 = QConv2d(quant_type, self.conv3, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu3 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv4 and relu4
        self.qconv4 = QConv2d(quant_type, self.conv4, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu4 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize second max pool
        self.qmax_pool2d2 = QMaxPooling2d(quant_type, 2, 2, ceil_mode=False, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv5 and relu5
        self.qconv5 = QConv2d(quant_type, self.conv5, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu5 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv6 and relu6
        self.qconv6 = QConv2d(quant_type, self.conv6, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu6 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv7 and relu7
        self.qconv7 = QConv2d(quant_type, self.conv7, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu7 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize third max pool with ceil_mode=True
        self.qmax_pool2d3 = QMaxPooling2d(quant_type, 2, 2, ceil_mode=True, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv8 and relu8
        self.qconv8 = QConv2d(quant_type, self.conv8, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu8 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv9 and relu9
        self.qconv9 = QConv2d(quant_type, self.conv9, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu9 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv10 and relu10
        self.qconv10 = QConv2d(quant_type, self.conv10, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu10 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)  #conv4-3

   
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmax_pool2d1(x)
    
        x = self.qconv3(x)
        x = self.qrelu3(x)
        x = self.qconv4(x)
        x = self.qrelu4(x)
        x = self.qmax_pool2d2(x)
    
        x = self.qconv5(x)
        x = self.qrelu5(x)
        x = self.qconv6(x)
        x = self.qrelu6(x)
        x = self.qconv7(x)
        x = self.qrelu7(x)
        x = self.qmax_pool2d3(x)
    
        x = self.qconv8(x)
        x = self.qrelu8(x)
        x = self.qconv9(x)
        x = self.qrelu9(x)
        x = self.qconv10(x)
        x = self.qrelu10(x)  #conv4-3
        return x,self.qconv10.qo

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmax_pool2d1.freeze(self.qconv2.qo)

        self.qconv3.freeze(qi=self.qconv2.qo)
        self.qrelu3.freeze(self.qconv3.qo)
        self.qconv4.freeze(qi=self.qconv3.qo)
        self.qrelu4.freeze(self.qconv4.qo)
        self.qmax_pool2d2.freeze(self.qconv4.qo)

        self.qconv5.freeze(qi=self.qconv4.qo)
        self.qrelu5.freeze(self.qconv5.qo)
        self.qconv6.freeze(qi=self.qconv5.qo)
        self.qrelu6.freeze(self.qconv6.qo)
        self.qconv7.freeze(qi=self.qconv6.qo)
        self.qrelu7.freeze(self.qconv7.qo)
        self.qmax_pool2d3.freeze(self.qconv7.qo)

        self.qconv8.freeze(qi=self.qconv7.qo)
        self.qrelu8.freeze(self.qconv8.qo)
        self.qconv9.freeze(qi=self.qconv8.qo)
        self.qrelu9.freeze(self.qconv9.qo)
        self.qconv10.freeze(qi=self.qconv9.qo)
        self.qrelu10.freeze(self.qconv10.qo)  #conv4-3

    def quantize_inference(self, x):
        qx = self.qconv1.quantize_inference(x)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmax_pool2d1.quantize_inference(qx)
    
        qx = self.qconv3.quantize_inference(qx)
        qx = self.qrelu3.quantize_inference(qx)
        qx = self.qconv4.quantize_inference(qx)
        qx = self.qrelu4.quantize_inference(qx)
        qx = self.qmax_pool2d2.quantize_inference(qx)
    
        qx = self.qconv5.quantize_inference(qx)
        qx = self.qrelu5.quantize_inference(qx)
        qx = self.qconv6.quantize_inference(qx)
        qx = self.qrelu6.quantize_inference(qx)
        qx = self.qconv7.quantize_inference(qx)
        qx = self.qrelu7.quantize_inference(qx)
        qx = self.qmax_pool2d3.quantize_inference(qx)
    
        qx = self.qconv8.quantize_inference(qx)
        qx = self.qrelu8.quantize_inference(qx)
        qx = self.qconv9.quantize_inference(qx)
        qx = self.qrelu9.quantize_inference(qx)
        qx = self.qconv10.quantize_inference(qx)
        qx = self.qrelu10.quantize_inference(qx)  #conv4-3
        return qx

# Vgg_upto_Conv7是vgg的Conv4_3后面的部分
class Vgg_upto_fc7(nn.Module):

    def __init__(self,pre_qo):
        super(Vgg_upto_fc7, self).__init__()
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv14 = nn.Conv2d(512, 1024, 3, 1, 6, dilation=6)
        self.conv15 = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.pre_qo = pre_qo

    def forward(self, x):
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, 3, 1, 1)
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        return x

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize fourth max pool
        self.qmax_pool2d4 = QMaxPooling2d(quant_type, 2, 2, ceil_mode=False, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv11 and relu11
        self.qconv11 = QConv2d(quant_type, self.conv11, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu11 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv12 and relu12
        self.qconv12 = QConv2d(quant_type, self.conv12, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu12 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv13 and relu13
        self.qconv13 = QConv2d(quant_type, self.conv13, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu13 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize fifth max pool with padding
        self.qmax_pool2d5 = QMaxPooling2d(quant_type, 3, 1, 1, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv14 and relu14
        self.qconv14 = QConv2d(quant_type, self.conv14, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu14 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)

        # Quantize conv15 and relu15
        self.qconv15 = QConv2d(quant_type, self.conv15, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu15 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qmax_pool2d4(x)
    
        x = self.qconv11(x)
        x = self.qrelu11(x)
        x = self.qconv12(x)
        x = self.qrelu12(x)
        x = self.qconv13(x)
        x = self.qrelu13(x)
        x = self.qmax_pool2d5(x)
    
        x = self.qconv14(x)
        x = self.qrelu14(x)
        x = self.qconv15(x)
        x = self.qrelu15(x)
    
        return x,self.qconv15.qo

    def freeze(self):
        self.qmax_pool2d4.freeze(self.pre_qo)

        self.qconv11.freeze(qi=self.pre_qo)
        self.qrelu11.freeze(self.qconv11.qo)
        self.qconv12.freeze(qi=self.qconv11.qo)
        self.qrelu12.freeze(self.qconv12.qo)
        self.qconv13.freeze(qi=self.qconv12.qo)
        self.qrelu13.freeze(self.qconv13.qo)
        self.qmax_pool2d5.freeze(self.qconv13.qo)

        self.qconv14.freeze(qi=self.qconv13.qo)
        self.qrelu14.freeze(self.qconv14.qo)
        self.qconv15.freeze(qi=self.qconv14.qo)
        self.qrelu15.freeze(self.qconv15.qo)

    def quantize_inference(self, x):
        qx = self.qmax_pool2d4.quantize_inference(x)
    
        qx = self.qconv11.quantize_inference(qx)
        qx = self.qrelu11.quantize_inference(qx)
        qx = self.qconv12.quantize_inference(qx)
        qx = self.qrelu12.quantize_inference(qx)
        qx = self.qconv13.quantize_inference(qx)
        qx = self.qrelu13.quantize_inference(qx)
        qx = self.qmax_pool2d5.quantize_inference(qx)
    
        qx = self.qconv14.quantize_inference(qx)
        qx = self.qrelu14.quantize_inference(qx)
        qx = self.qconv15.quantize_inference(qx)
        qx = self.qrelu15.quantize_inference(qx)
    
        return qx
    

class Extra_Conv6_2(nn.Module):

    def __init__(self,pre_qo):
        super(Extra_Conv6_2, self).__init__()
        self.conv16 = nn.Conv2d(1024, 256, 1, 1)
        self.conv17 = nn.Conv2d(256, 512, 3, 2 ,1)
        self.pre_qo = pre_qo

    def forward(self, x):
        x = F.relu(self.conv16(x)) 
        x = F.relu(self.conv17(x))
        return x

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv16 and Conv17
        self.qconv16 = QConv2d(quant_type, self.conv16, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu16 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
        self.qconv17 = QConv2d(quant_type, self.conv17, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu17 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qconv16(x)
        x = self.qrelu16(x)
        x = self.qconv17(x)
        x = self.qrelu17(x)
        return x,self.qconv17.qo

    def freeze(self):
        self.qconv16.freeze(qi=self.pre_qo)
        self.qrelu16.freeze(self.qconv16.qo)
        self.qconv17.freeze(self.qconv16.qo)
        self.qrelu17.freeze(self.qconv17.qo)

    def quantize_inference(self, x):
        qx = self.qconv16.quantize_inference(x)
        qx = self.qrelu16.quantize_inference(qx)
        qx = self.qconv17.quantize_inference(qx)
        qx = self.qrelu17.quantize_inference(qx)
        return qx
        

class Extra_Conv7_2(nn.Module):

    def __init__(self,pre_qo):
        super(Extra_Conv7_2, self).__init__()
        self.conv18 = nn.Conv2d(512, 128, 1, 1)
        self.conv19 = nn.Conv2d(128, 256, 3, 2 ,1)
        self.pre_qo = pre_qo

    def forward(self, x):
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        return x

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv16 and Conv17
        self.qconv18 = QConv2d(quant_type, self.conv18, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu18 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
        self.qconv19 = QConv2d(quant_type, self.conv19, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu19 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qconv18(x)
        x = self.qrelu18(x)
        x = self.qconv19(x)
        x = self.qrelu19(x)
        return x,self.qconv19.qo

    def freeze(self):
        self.qconv18.freeze(qi=self.pre_qo)
        self.qrelu18.freeze(self.qconv18.qo)
        self.qconv19.freeze(self.qconv18.qo)
        self.qrelu19.freeze(self.qconv19.qo)

    def quantize_inference(self, x):
        qx = self.qconv18.quantize_inference(x)
        qx = self.qrelu18.quantize_inference(qx)
        qx = self.qconv19.quantize_inference(qx)
        qx = self.qrelu19.quantize_inference(qx)
        return qx
    
    
class Extra_Conv8_2(nn.Module):

    def __init__(self,pre_qo):
        super(Extra_Conv8_2, self).__init__()
        self.conv20 = nn.Conv2d(256, 128, 1, 1)
        self.conv21 = nn.Conv2d(128, 256, 3, 1)
        self.pre_qo = pre_qo

    def forward(self, x):
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))
        return x

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv16 and Conv17
        self.qconv20 = QConv2d(quant_type, self.conv20, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu20= QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
        self.qconv21 = QConv2d(quant_type, self.conv21, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu21 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qconv20(x)
        x = self.qrelu20(x)
        x = self.qconv21(x)
        x = self.qrelu21(x)
        return x, self.qconv21.qo

    def freeze(self):
        self.qconv20.freeze(qi=self.pre_qo)
        self.qrelu20.freeze(self.qconv20.qo)
        self.qconv21.freeze(self.qconv20.qo)
        self.qrelu21.freeze(self.qconv21.qo)

    def quantize_inference(self, x):
        qx = self.qconv20.quantize_inference(x)
        qx = self.qrelu20.quantize_inference(qx)
        qx = self.qconv21.quantize_inference(qx)
        qx = self.qrelu21.quantize_inference(qx)
        return qx

    
class Extra_Conv9_2(nn.Module):

    def __init__(self,pre_qo):
        super(Extra_Conv9_2, self).__init__()
        self.conv22 = nn.Conv2d(256, 128, 1, 1)
        self.conv23 = nn.Conv2d(128, 256, 3, 1)
        self.pre_qo = pre_qo

    def forward(self, x):
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        return x

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv16 and Conv17
        self.qconv22 = QConv2d(quant_type, self.conv22, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu22 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
        self.qconv23 = QConv2d(quant_type, self.conv23, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu23 = QReLU(quant_type, qi=False, qo=False, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qconv22(x)
        x = self.qrelu22(x)
        x = self.qconv23(x)
        x = self.qrelu23(x)
        return x,self.qconv23.qo

    def freeze(self):
        self.qconv22.freeze(qi=self.pre_qo)
        self.qrelu22.freeze(self.qconv22.qo)
        self.qconv23.freeze(self.qconv22.qo)
        self.qrelu23.freeze(self.qconv23.qo)

    def quantize_inference(self, x):
        qx = self.qconv22.quantize_inference(x)
        qx = self.qrelu22.quantize_inference(qx)
        qx = self.qconv23.quantize_inference(qx)
        qx = self.qrelu23.quantize_inference(qx)
        return qx

# Detector, qi_flag的作用是区分第一个Detector和其它，第一个经过了L2Norm，feature1作为输入其范围已经变了，需要初始化qi
# Detector1: qi_flag = True, Other: qi_flag = False
class Detector(nn.Module):

    def __init__(self,pre_qo,qi_flag,in_channels,num_anchors,num_classes):
        super(Detector, self).__init__()
        self.pre_qo = pre_qo
        self.qi_flag = qi_flag
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv_loc = nn.Conv2d(self.in_channels, self.num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_loc(x)
        return x 

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv16 and Conv17
        self.qconv_loc = QConv2d(quant_type, self.conv_loc, qi=self.qi_flag, qo=True, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qconv_loc(x)
        return x

    def freeze(self):
        if self.qi_flag:
            self.qconv_loc.freeze()
        else:
            self.qconv_loc.freeze(qi=self.pre_qo)

    def quantize_inference(self, x):
        qx = self.qconv_loc.quantize_inference(x)
        return qx
    
# Classifier
class Classifier(nn.Module):

    def __init__(self,pre_qo,qi_flag,in_channels,num_anchors,num_classes):
        super(Classifier, self).__init__()
        self.pre_qo = pre_qo
        self.qi_flag = qi_flag
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv_conf = nn.Conv2d(self.in_channels, self.num_anchors*self.num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_conf(x)
        return x 

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        # Quantize conv16 and Conv17
        self.qconv_conf = QConv2d(quant_type, self.conv_conf, qi=self.qi_flag, qo=True, num_bits=num_bits, e_bits=e_bits)
    
    def quantize_forward(self, x):
        x = self.qconv_conf(x)
        return x

    def freeze(self):
        if self.qi_flag:
            self.qconv_conf.freeze()
        else:
            self.qconv_conf.freeze(qi=self.pre_qo)

    def quantize_inference(self, x):
        qx = self.qconv_conf.quantize_inference(x)
        return qx
    

# SSD
class SSD(nn.Module):

    def __init__(self,num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.L2Norm = L2Norm(512,20)
        self.cfg = (coco, voc)[num_classes == 21]
        #phase == test
        self.softmax = nn.Softmax(dim=-1)
        self.detect = DetectModule(num_classes, 0, 200, 0.01, 0.45)

        self.priors = PriorBoxFunction.apply(self.cfg)

        # Buils layers
        self.vgg_upto_conv4_3 = Vgg_upto_Conv4_3()
        self.vgg_upto_fc7 = Vgg_upto_fc7(pre_qo=None)
        self.extra_conv6_2 = Extra_Conv6_2(pre_qo=None)
        self.extra_conv7_2 = Extra_Conv7_2(pre_qo=None)
        self.extra_conv8_2 = Extra_Conv8_2(pre_qo=None)
        self.extra_conv9_2 = Extra_Conv9_2(pre_qo=None)

        self.loc_layers = nn.ModuleList([
            Detector(pre_qo=None, qi_flag = True, in_channels=512, num_anchors=4, num_classes=self.num_classes),
            Detector(pre_qo=None, qi_flag = False, in_channels=1024, num_anchors=6, num_classes=self.num_classes),
            Detector(pre_qo=None, qi_flag = False,  in_channels=512, num_anchors=6, num_classes=self.num_classes),
            Detector(pre_qo=None, qi_flag = False, in_channels=256, num_anchors=6, num_classes=self.num_classes),
            Detector(pre_qo=None, qi_flag = False, in_channels=256, num_anchors=4, num_classes=self.num_classes),
            Detector(pre_qo=None, qi_flag = False, in_channels=256, num_anchors=4, num_classes=self.num_classes)
        ])

        self.conf_layers = nn.ModuleList([
           Classifier(pre_qo=None, qi_flag = True, in_channels=512, num_anchors=4, num_classes=self.num_classes),
           Classifier(pre_qo=None, qi_flag = False, in_channels=1024, num_anchors=6, num_classes=self.num_classes),
           Classifier(pre_qo=None, qi_flag = False, in_channels=512, num_anchors=6, num_classes=self.num_classes),
           Classifier(pre_qo=None, qi_flag = False, in_channels=256, num_anchors=6, num_classes=self.num_classes),
           Classifier(pre_qo=None, qi_flag = False, in_channels=256, num_anchors=4, num_classes=self.num_classes),
           Classifier(pre_qo=None, qi_flag = False, in_channels=256, num_anchors=4, num_classes=self.num_classes)
        ])        


    def forward(self, x):
        conv4_3_feats = self.vgg_upto_conv4_3(x)
        normed_conv4_3_feats = self.L2Norm(conv4_3_feats)
        fc7_feats = self.vgg_upto_fc7(conv4_3_feats)

        # Extra feature layers
        conv6_2_feats = self.extra_conv6_2(fc7_feats)
        conv7_2_feats = self.extra_conv7_2(conv6_2_feats)
        conv8_2_feats = self.extra_conv8_2(conv7_2_feats)
        conv9_2_feats = self.extra_conv9_2(conv8_2_feats)

        # Detector and classifier layers
        loc_preds = list()
        conf_preds = list()
        features = [normed_conv4_3_feats, fc7_feats, conv6_2_feats, conv7_2_feats, conv8_2_feats, conv9_2_feats]
        for i, feature in enumerate(features):
            loc_pred = self.loc_layers[i](feature).permute(0, 2, 3, 1).contiguous()
            conf_pred = self.conf_layers[i](feature).permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred)
            conf_preds.append(conf_pred)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)

        output = self.detect(
            loc.view(loc.size(0), -1, 4),
            self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
            self.priors.type_as(x)
        )

        return output 

    def quantize(self, quant_type='INT', num_bits=8, e_bits=3):
        self.vgg_upto_conv4_3.quantize(quant_type, num_bits, e_bits)
        self.vgg_upto_fc7.quantize(quant_type, num_bits, e_bits)
        self.extra_conv6_2.quantize(quant_type, num_bits, e_bits)
        self.extra_conv7_2.quantize(quant_type, num_bits, e_bits)
        self.extra_conv8_2.quantize(quant_type, num_bits, e_bits)
        self.extra_conv9_2.quantize(quant_type, num_bits, e_bits)
        for layer in self.loc_layers:
            layer.quantize(quant_type, num_bits, e_bits)
        for layer in self.conf_layers:
            layer.quantize(quant_type, num_bits, e_bits)
    
    def quantize_forward(self, x):
        fq_conv4_3_feats, qo = self.vgg_upto_conv4_3.quantize_forward(x)
        self.vgg_upto_fc7.pre_qo = qo
        self.loc_layers[0].pre_qo = qo
        self.conf_layers[0].pre_qo = qo
        normed_fq_conv4_3_feats = self.L2Norm(fq_conv4_3_feats)
        self.loc_layers[0].quantize_forward(normed_fq_conv4_3_feats)
        self.conf_layers[0].quantize_forward(normed_fq_conv4_3_feats)

        fq_fc7_feats, qo = self.vgg_upto_fc7.quantize_forward(fq_conv4_3_feats)
        self.extra_conv6_2.pre_qo = qo
        self.loc_layers[1].pre_qo = qo
        self.conf_layers[1].pre_qo = qo
        self.loc_layers[1].quantize_forward(fq_fc7_feats)
        self.conf_layers[1].quantize_forward(fq_fc7_feats)

        fq_conv6_2_feats, qo = self.extra_conv6_2.quantize_forward(fq_fc7_feats)
        self.extra_conv7_2.pre_qo = qo
        self.loc_layers[2].pre_qo = qo
        self.conf_layers[2].pre_qo = qo
        self.loc_layers[2].quantize_forward(fq_conv6_2_feats)
        self.conf_layers[2].quantize_forward(fq_conv6_2_feats)

        fq_conv7_2_feats, qo = self.extra_conv7_2.quantize_forward(fq_conv6_2_feats)
        self.extra_conv8_2.pre_qo = qo
        self.loc_layers[3].pre_qo = qo
        self.conf_layers[3].pre_qo = qo
        self.loc_layers[3].quantize_forward(fq_conv7_2_feats)
        self.conf_layers[3].quantize_forward(fq_conv7_2_feats)

        fq_conv8_2_feats, qo = self.extra_conv8_2.quantize_forward(fq_conv7_2_feats)
        self.extra_conv9_2.pre_qo = qo
        self.loc_layers[4].pre_qo = qo
        self.conf_layers[4].pre_qo = qo
        self.loc_layers[4].quantize_forward(fq_conv8_2_feats)
        self.conf_layers[4].quantize_forward(fq_conv8_2_feats)

        fq_conv9_2_feats, qo = self.extra_conv9_2.quantize_forward(fq_conv8_2_feats)
        self.loc_layers[5].pre_qo = qo
        self.conf_layers[5].pre_qo = qo
        self.loc_layers[5].quantize_forward(fq_conv9_2_feats)
        self.conf_layers[5].quantize_forward(fq_conv9_2_feats)

    def freeze(self):
        self.vgg_upto_conv4_3.freeze()
        self.vgg_upto_fc7.freeze()
        self.extra_conv6_2.freeze()
        self.extra_conv7_2.freeze()
        self.extra_conv8_2.freeze()
        self.extra_conv9_2.freeze()
        
        '''
        self.loc_layers[0].freeze()
        self.loc_layers[1].freeze()
        self.loc_layers[2].freeze()
        self.loc_layers[3].freeze()
        self.loc_layers[4].freeze()
        self.loc_layers[5].freeze()
        '''
        for layer in self.loc_layers:
            layer.freeze()

        for layer in self.conf_layers:
            layer.freeze()

    def quantize_inference(self, x):
        qconv4_3_feats = self.vgg_upto_conv4_3.quantize_inference(x)
        normed_qconv4_3_feats = self.L2Norm(qconv4_3_feats)
        qfc7_feats = self.vgg_upto_fc7.quantize_inference(qconv4_3_feats)
        qconv6_2_feats = self.extra_conv6_2.quantize_inference(qfc7_feats)
        qconv7_2_feats = self.extra_conv7_2.quantize_inference(qconv6_2_feats)
        qconv8_2_feats = self.extra_conv8_2.quantize_inference(qconv7_2_feats)
        qconv9_2_feats = self.extra_conv9_2.quantize_inference(qconv8_2_feats)

        loc_preds = []
        conf_preds = []
        features = [normed_qconv4_3_feats, qfc7_feats, qconv6_2_feats, qconv7_2_feats, qconv8_2_feats, qconv9_2_feats]

        for i, feature in enumerate(features):
            loc_pred = self.loc_layers[i].quantize_inference(feature).permute(0, 2, 3, 1).contiguous()
            cls_pred = self.conf_layers[i].quantize_inference(feature).permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred)
            conf_preds.append(cls_pred)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)

        output = self.detect(
            loc.view(loc.size(0), -1, 4),
            self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
            self.priors.type_as(x)
        )

        return output 