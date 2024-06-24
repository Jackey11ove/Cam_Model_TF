import torch
from yolov5_model import Yolo_v5

def main():
    
    # Quantization
    quant_model = Yolo_v5()
    # 打印量化模型的参数名称
    print("Quantized Model State Dict:")
    for key in quant_model.state_dict().keys():
       print(key)


if __name__ == "__main__":
    main()