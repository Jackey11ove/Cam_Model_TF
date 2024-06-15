import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ssd_model import SSD

def main():
    """
    full_precision_model_path = '../weights/VOC.pth'
    # Load full precision model's state_dict
    full_precision_state_dict = torch.load(full_precision_model_path)

    # Print full precision model's state_dict keys
    print("Full Precision Model State Dict:")
    for key in full_precision_state_dict.keys():
        print(key)
    """
    
    # Quantization
    quant_model = SSD(num_classes=21)
    # 打印量化模型的参数名称
    print("Quantized Model State Dict:")
    for key in quant_model.state_dict().keys():
       print(key)


if __name__ == "__main__":
    main()