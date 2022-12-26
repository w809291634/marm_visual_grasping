#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import cv2
import time
import random
import numpy as np
from rknnlite.api import RKNNLite
import os
"""
RK3588 yolov5s交通标志模型
"""
dir_f = os.path.abspath(os.path.dirname(__file__))
class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = self.get_max_scale(img, max_w, max_h)
        self._new_size = self.get_new_size(img, self.scale)
        self.__new_img = None

    def get_max_scale(self, img, max_w, max_h):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1)
        return scale

    def get_new_size(self, img, scale):
        return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    
    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img

class YoloV5RKNNDetector:
    def __init__(self, 
                 model_name, 
                 wh=(640, 640), 
                 masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 
                 anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
                 ):
        self.wh = wh
        self._masks = masks
        self._anchors = anchors
        self.path = os.path.join(dir_f,model_name,model_name)
        self.name_path=self.path +'.names'
        self.model_path=self.path +'.rknn'
        self._rknn = self.load_rknn_model(self.model_path)
        self.names=[]
        txt = open(self.name_path, "r", encoding="utf-8-sig")       #从names文件中读取标签
        for line in txt.readlines():
            line = line.strip()         # 去掉每行头尾空白
            if len(line) >0:
                self.names.append(str(line))
        print(self.names)
        self.draw_box = True

    def getClassIdx(self, cla_name):
        if cla_name in self.names:
            return self.names.index(cla_name)
        return -1

    def load_rknn_model(self,PATH):
        rknn = RKNNLite()
        ret = rknn.load_rknn(PATH)
        if ret != 0:
            print('load rknn model failed')
            exit(ret)
        print('done')
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)     #自动调度NPU,有三个NPU
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        return rknn

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self,boxes, box_confidences, box_class_probs, conf_thres):
        box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
        box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
        box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
        pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
        # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]
        return boxes, classes, scores

    def nms_boxes(self,boxes, scores, iou_thres):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None, score=None):
        tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            text=label+' score:%.2f'% score     #输出文本
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def _predict2json(self, img_src, _img, gain, conf_thres=0.4, iou_thres=0.45):
        # 推理返回的结构
        respond={
        "code" : None,          # 返回码
        "msg" : None,           # 返回消息
        "result" :              # 返回结果
            {
                "obj_list" : 
                [
                    {"location" : {"left": None,"top": None,"right": None, "bottom": None},"name": None,"score": None } # 目标对应的位置、名称、置信值
                ],
                "obj_num" : 1,
                "time" : None           # 推理时间
            }, 
        }
        # 准备推理
        src_h, src_w = img_src.shape[:2]
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        # 调用NPU进行推理
        pred_onx = self._rknn.inference(inputs=[_img])
        respond["result"]["time"]=round((time.time() - t0)*1000,2)

        # 处理推理结果
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = self.sigmoid(pred_onx[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = self.filter_boxes(box, box_confidence, box_class_probs, conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s, iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            respond["code"]=201
            respond["msg"]="NO_OBJECT"
            respond["result"]["obj_list"]=[]
            respond["result"]["obj_num"]=0
            return respond
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        # 返回推理的结果
        respond["code"]=200
        respond["msg"]="SUCCESS"
        respond["result"]["obj_list"]=[]
        respond["result"]["obj_num"]=len(nboxes)
        
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            obj={}
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))

            obj["location"]={"left":int(x1),"top":int(y1),"right":int(x2),"bottom":int(y2)}
            obj["name"]=self.names[cl]
            obj["score"]=float(score)
            respond["result"]["obj_list"].append(obj)

            if self.draw_box:
                self.plot_one_box((x1, y1, x2, y2), img_src, label=self.names[cl],score=score)
        return respond

    def _predict(self, img_src, _img, gain, conf_thres=0.4, iou_thres=0.45):
        # 推理返回的结构
        rets = []
        types = []
        pp = []
        # 准备推理
        src_h, src_w = img_src.shape[:2]
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        # 调用NPU进行推理
        pred_onx = self._rknn.inference(inputs=[_img])

        # 处理推理结果
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = self.sigmoid(pred_onx[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh           # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)     # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = self.filter_boxes(box, box_confidence, box_class_probs, conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        # 后处理
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s, iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            rets = []
            types = []
            pp = []
            return (img_src, rets, types, pp)
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        
        # 分组打包数据并获取
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            score=float(score)              #获取此组分值
            if score > 0.8:  
                x *= gain[0]
                y *= gain[1]
                w *= gain[0]
                h *= gain[1]
                x1 = max(0, np.floor(x).astype(int))
                y1 = max(0, np.floor(y).astype(int))
                x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
                y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
                # 处理结果
                _rect = [int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)]
                rets.append(_rect)
                types.append(self.names[cl])
                pp.append(score)
                self.plot_one_box((x1, y1, x2, y2), img_src, color=(0, 0, 255), label=self.names[cl],score=score)
        return (img_src, rets, types, pp)

    def predict_resize(self, img_src,  conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片,预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain, conf_thres, iou_thres, )

    def letterbox(self,img, new_wh=(416, 416), color=(114, 114, 114)):
        a = AutoScale(img, *new_wh)
        new_img = a.new_img
        h, w = new_img.shape[:2]
        new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
        return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale)

    def predict(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        _img, gain = self.letterbox(img_src, self.wh)
        return self._predict(img_src, _img, gain, conf_thres, iou_thres)

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

if __name__ == '__main__':
    detector = YoloV5RKNNDetector("wooden_medicine")         #初始化yolov5 RKNN检测类  wooden_medicine:检测药盒  traffic_lights:检测交通灯
    cam = cv2.VideoCapture(7)
    if cam.isOpened() == False:
        print('Camera read error, exit')
        exit(1)
    cam.set(cv2.CAP_PROP_FPS, 30) 
    while True:
        ret, frame = cam.read()
        #labels, boxes = detector.predict(frame)
        ret = detector.predict_resize(frame)               #返回具体的检测结果，frame为处理后的图像
        print(ret[1],ret[2],ret[3])
        # 获取视频fps
        #fps = cam.get(cv2.CAP_PROP_FPS)
        '''
        if len(labels) > 0:
            print('labels:', labels)
            print('boxes:', boxes)
        else:
            print('no results')
        '''
        cv2.namedWindow('result', 10)
        cv2.resizeWindow('result', 640, 480)
        cv2.imshow('result', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        #print('FPS:', fps)

    cam.release()
    cv2.destroyAllWindows()

