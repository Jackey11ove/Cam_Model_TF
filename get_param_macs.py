import sys
import torch
import torchvision.models as models
from ptflops import get_model_complexity_info

from ssd.ssd_model import SSD
from yolo_v5.yolov5_model import Yolo_v5


if __name__ == "__main__":
    model_name = sys.argv[1]
    dataset = sys.argv[2]
    datasize = int(sys.argv[3])
    classnum = int(sys.argv[4])
    #model = Model(model_name,dataset,datasize,classnum)

    # full_file = 'ckpt/cifar10_'+model_name+'.pt'
    # model.load_state_dict(torch.load(full_file))
    if model_name == 'SSD' or dataset == 'voc2007' or datasize == 300:
        model = SSD(classnum)
        macs, params = get_model_complexity_info(model, (3, 300, 300), as_strings=True, print_per_layer_stat=True)
    elif model_name == 'Yolo_v5' or dataset == 'voc0712test' or datasize == 640:
        model = Yolo_v5()
        macs, params = get_model_complexity_info(model, (3, 640, 640), as_strings=True, print_per_layer_stat=True)
    else:
        assert False, "invalid data set"
