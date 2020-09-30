import os
import cv2
import json
import time
import uuid
import base64
import web
from PIL import Image
import model
from config import DETECTANGLE
from apphelper.image import union_rbox, adjust_box_to_origin
from application import trainTicket, idcard

def textLine(image):
    """
    单行识别
    :param image: 图片路径
    :return:
    """
    img = cv2.imread(image)  ##GBR
    H, W = img.shape[:2]
    partImg = Image.fromarray(img)
    text = model.crnnOcr(partImg.convert('L'))
    res = [{'text': text, 'name': '0', 'box': [0, 0, W, 0, W, H, 0, H]}]
    print(res)
    return res

def angle_recoginse(image, detectAngle):
    """
    角度识别
    :param image:
    :param detectAngle: bool值，是否检测文字方向
    :return:
    """
    img = cv2.imread(image)  ##GBR
    H, W = img.shape[:2]
    _, result, angle = model.model(img,
                                   detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
                                               ),
                                   leftAdjust=True,  ##对检测的文本行进行向左延伸
                                   rightAdjust=True,  ##对检测的文本行进行向右延伸
                                   alph=0.01,  ##对检测的文本行进行向右、左延伸的倍数
                                   )
    return result, angle

def normal_ocr(image):
    """
    通用识别
    :param image:
    :return:
    """
    img = cv2.imread(image)  ##GBR
    H, W = img.shape[:2]
    ang_res, angle = angle_recoginse(image=image, detectAngle=True)
    result = union_rbox(ang_res, 0.2)
    res = [{'text': x['text'],
            'name': str(i),
            'box': {'cx': x['cx'],
                    'cy': x['cy'],
                    'w': x['w'],
                    'h': x['h'],
                    'angle': x['degree']

                    }
            } for i, x in enumerate(result)]
    res = adjust_box_to_origin(img, angle, res)  ##修正box
    return res
def train_ocr(image):
    """
    火车票识别
    :param image:
    :return:
    """
    img = cv2.imread(image)  ##GBR
    H, W = img.shape[:2]
    ang_res, angle = angle_recoginse(image=image, detectAngle=True)
    res = trainTicket.trainTicket(ang_res)
    res = res.res
    res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
    return res
def idcard_ocr(image):
    """
    身份证
    :param image:
    :return:
    """
    img = cv2.imread(image)  ##GBR
    H, W = img.shape[:2]
    ang_res, angle = angle_recoginse(image=image, detectAngle=True)
    res = idcard.idcard(ang_res)
    res = res.res
    res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
    return res



if __name__ == '__main__':
    res = normal_ocr(image="test/img.jpeg")
    print(res)
