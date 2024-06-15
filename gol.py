# -*- coding: utf-8 -*-

# 用于多个module之间共享全局变量
def _init():  # 初始化
    global _global_dict
    _global_dict = {}
 
def set_value(value,is_bias=False):
    # 定义一个全局变量
    if is_bias:
        _global_dict[0] = value
    else:
        _global_dict[1] = value
 
 
def get_value(is_bias=False): # 给bias独立于各变量外的精度
    if is_bias:
        return _global_dict[0]
    else:
        return _global_dict[1]  

