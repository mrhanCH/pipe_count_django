from django.shortcuts import HttpResponse
from pipeCount.pipe import findPipes

import os, json


# 主要API接口
def findPipe(request):
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
    data = {'resultCount': len(result_arr), 'width': width,
            'height': height, 'picResult': result_arr, 'name': str(image)[:-4]}
    response = {'code': 200, 'msg': '查询成功', 'data': data}
    return HttpResponse(json.dumps(response))


# 保存结果生成YOLOV5格式数据集文件
def saveResult(request):
    if request.method == 'POST':
        parames = json.loads(request.body.decode("utf-8"))
        allport = parames['allPort']
        name = parames['name']
        pWidth = parames['pWidth']
        pHeight = parames['pHeight']
        if name != '':
            # 文件名与前端上传的图片名称保持一致
            txt = 'pipeCount/static/output/' + str(name) + '.txt'
            Note = open(txt, mode='w', encoding='utf-8')
            for item in allport:
                # 转换格式
                box = convert((pWidth, pHeight), item)
                # 写入txt，共5个字段
                Note.write("%s %s %s %s %s\n" % (
                    0, box[0], box[1], box[2], box[3]))
            Note.close()
        # 回调json格式
        response = {'code': 200, 'msg': '保存成功'}
        return HttpResponse(json.dumps(response))
    else:
        return HttpResponse(json.dumps({'code': 100, 'msg': '非法请求'}))


# 保存客户端上传图片到本地
def saveImage(image, filename):
    if not os.path.exists(filename):  # 如何不为空
        with open(filename, 'wb')as f:  # 转为二进制写入
            f.write(image.read())
            f.closed
    return filename


def convert(size, box):
    '''
    size: 图片的宽和高(w,h)
    box格式: x,y,w,h
    返回值：x_center/image_width y_center/image_height width/image_width height/image_height
    '''
    dw = 1. / float(size[0])
    dh = 1. / float(size[1])
    x = float(box['leftTop']['xaxis'] + (box['rightDown']['xaxis'] - box['leftTop']['xaxis'])) / 2.0
    y = float(box['leftTop']['yaxis'] + (box['rightDown']['yaxis'] - box['leftTop']['yaxis'])) / 2.0
    w = float(box['rightDown']['xaxis'] - box['leftTop']['xaxis'])
    h = float(box['rightDown']['yaxis'] - box['leftTop']['yaxis'])

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
