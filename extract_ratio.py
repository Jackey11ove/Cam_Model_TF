import sys
import os


# 从get_param.py输出重定向文件val.txt中提取参数量和计算量
def extract_ratio(model_name,dataset):
    fr = open('param_macs/'+dataset+'/'+model_name+'.txt','r')
    lines = fr.readlines()
    
    #跳过warning
    for i in range(len(lines)):
        if 'SSD' in lines[i] or 'Yolo_v5' in lines[i]:
            head = i+1
            break 
    Mac = lines[head].split('Mac,')[0].split(',')[-1]
    if 'M' in Mac:
        Mac = Mac.split('M')[0]
        Mac = float(Mac)
    elif 'G' in Mac:
        Mac = Mac.split('G')[0]
        Mac = float(Mac)
        Mac *= 1024
    
    Param = lines[head].split(',')[0]
    if 'M' in Param:
        Param = Param.split('M')[0]
        Param = float(Param)
    elif 'k' in Param:
        Param = Param.split('k')[0]
        Param = float(Param)
        Param /= 1024
    
    layer = []
    par_ratio = []
    flop_ratio = []
    for line in lines:
        if '(' in line and ')' in line:
            layer.append(line.split(')')[0].split('(')[1])
            r1 = line.split('%')[0].split(',')[-1]
            r1 = float(r1)
            par_ratio.append(r1)
            r2 = line.split('%')[-2].split(',')[-1]
            r2 = float(r2)
            flop_ratio.append(r2)

    return Mac, Param, layer, par_ratio, flop_ratio