from django.shortcuts import render
from django.shortcuts import HttpResponse
from pipeCount.pipe import findPipes

import sys, os, time, json

Basedir = os.path.dirname(__file__)  # 获取当前路径的上一级 极为s1


# Create your views here.
def index(request):
    # YOLOV5_P6训练模型，回调:原始图片宽，高，[左上坐标，右下坐标，中心点坐标，范围参数]
    # findPipes参数（onnx模型地址-默认为空使用自带模型，识别图片地址，是否保存识别结果）
    width, height, result_arr = findPipes('', 'pipeCount/static/image/gangguan.jpg', True)
    data = {}
    data['width'] = width
    data['height'] = height
    data['result_arr'] = result_arr
    return HttpResponse(json.dumps(data))
