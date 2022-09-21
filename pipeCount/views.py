from django.shortcuts import render
from django.shortcuts import HttpResponse
from pipeCount.pipe import findPipes

import sys, os, time, json

# Basedir = os.path.dirname(__file__)  # 获取当前路径的上一级 极为s1


# 主要API接口
def index(request):
    # 获取前端上传的图片
    image = request.FILES['image']
    if image is None:
        return HttpResponse(json.dumps({'code': 100, 'msg': '请上传图片'}))
    # 保存图片到本地
    local_image = saveImage(image, 'pipeCount/static/image/' + str(image))
    # YOLOV5_P6训练模型，回调:原始图片宽，高，[左上坐标，右下坐标，中心点坐标，范围参数]
    # findPipes参数（onnx模型地址-默认为空使用自带模型，识别图片地址，是否保存识别结果）
    width, height, result_arr = findPipes('', local_image, True)
    # 回调json格式
    data = {'recognitionCount': len(result_arr), 'width': width,
            'height': height, 'picRecognizeds': result_arr}
    response = {'code': 200, 'msg': '查询成功', 'data': data}
    return HttpResponse(json.dumps(response))


# 保存客户端上传图片到本地
def saveImage(image, filename):
    if not os.path.exists(filename):  # 如何不为空
        with open(filename, 'wb')as f:  # 转为二进制写入
            f.write(image.read())
            f.closed
    return filename
