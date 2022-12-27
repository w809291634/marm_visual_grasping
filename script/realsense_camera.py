#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import os
import sys
import tf         
import cvwin
import cv2
import time
import numpy as np
from collections import OrderedDict
import pyrealsense2 as rs
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Pose
import yaml

this = sys.modules[__name__]
this.config_path="/home/zonesion/catkin_ws/src/marm_controller/config/config.yaml"

this.dir_f = os.path.abspath(os.path.dirname(__file__))

with open(this.config_path, "r") as f:
  if sys.version_info < (3, 0):
    config = yaml.load(f.read())
  else:    
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

class RealsenseCamera(object):
    def __init__(self, MARGIN_PIX=7):
      from obj_detection_rk3588 import detection
      self.findObj = detection.YoloV5RKNNDetector("wooden_medicine")  #初始化目标检测类
      self.MARGIN_PIX = MARGIN_PIX
      self.window_name='camera'
      self.camera_link="camera_link"
      self.open_wins=[]
      self.camera_init(424,240)
      self.tf_listener = tf.TransformListener()
      self.tf_listener.waitForTransform("base_link", self.camera_link, rospy.Time(0), rospy.Duration(1))

    def camera_init(self,WIDTH,HEIGHT):
      self.pipeline = rs.pipeline()
      config = rs.config()
      config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)     #使能深度相机
      config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)    #使能彩色相机
      try:
        # if self.__camera_enable!=True:
          self.profile = self.pipeline.start(config)
          # self.__camera_enable=True

          # 保存相机内参
          frames = self.pipeline.wait_for_frames()            #等待相机坐标系生成
          color_frame = frames.get_color_frame()         #获取彩色相机坐标系
          self.intr = color_frame.profile.as_video_stream_profile().intrinsics
          camera_parameters = {'fx': self.intr.fx, 'fy': self.intr.fy,
                              'ppx': self.intr.ppx, 'ppy': self.intr.ppy,
                              'height': self.intr.height, 'width': self.intr.width,
                              'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                              }
          # print(camera_parameters)
          # 保存深度参数
          align_to = rs.stream.color                              #统一对齐到彩色相机
          align = rs.align(align_to)
          aligned_frames = align.process(frames)                  #对齐后的相机坐标系
          aligned_depth_frame = aligned_frames.get_depth_frame()  #对齐到彩色相机后的深度相机坐标系
          self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
      except RuntimeError as e:
        # if "UVC device is already opened!" in e.args[0]:
        #   self.stopCamera()
        print(e)
        sys.exit()
      # print(self.depth_intrin)

    def camera_data(self):
      # 图像对齐
      frames = self.pipeline.wait_for_frames()            #等待相机坐标系生成

      self.align_to = rs.stream.color                     #统一对齐到彩色相机
      self.align = rs.align(self.align_to)
      self.aligned_frames = self.align.process(frames)    #对齐后的相机坐标系

      self.aligned_depth_frame = self.aligned_frames.get_depth_frame()        #对齐到彩色相机后的深度相机坐标系
      self.color_frame = self.aligned_frames.get_color_frame()                #彩色相机坐标系

      if self.aligned_depth_frame and self.color_frame:
          self.color_data = np.asanyarray(self.color_frame.get_data())        #/camera/color/image_raw
          self.dep_data= np.asanyarray(self.aligned_depth_frame.get_data())   #/camera/aligned_depth_to_color/image_raw

    def max_word(self,lt):
      # 定义一个字典，用于保存每个元素及出现的次数
      d = {}
      # 记录做大的元素(字典的键)
      max_key = None
      for w in lt:
        if w not in d:
          # 统计该元素在列表中出现的次数
          count = lt.count(w)
          # 以元素作为键，次数作为值，保存到字典中
          d[w] = count
          # 记录最大元素
          if d.get(max_key, 0) < count:
            max_key = w
      return max_key,d

    def Depth_data(self,pos,hp,wp,minDeep=135,maxDeep=1000,range_key=2,grade=0.9):  #获取位置坐标系下的深度数据
        '''
        pos[0]  x坐标，单位pixels
        pos[1]  y坐标，单位pixels
        hp      纵向公差
        wp      横向公差
        minDeep 最小深度
        '''
        depimg = self.dep_data
        xx= pos[0]      #x坐标转存，单位pixels
        yy= pos[1]      #y坐标转存，单位pixels
        sumx = sumy = sumz = num =0
        list_deep=[]
        # dis = aligned_depth_frame.get_distance(x, y)      # 获取深度的接口
        for m in range(int(yy-hp), int(yy+hp)):             # 以yy中心，hp为公差的范围数组
            for n in range(int(xx-wp), int(xx+wp)):
                if depimg[m][n] < minDeep:
                    continue
                if depimg[m][n] > maxDeep:
                    continue
                list_deep.append(depimg[m][n])
        if(list_deep!=[]):
            max_length=(2*hp)*(2*wp)
            length=len(list_deep)               #获取深度数据的长度，长度不一定
            max_key,d=self.max_word(list_deep)
            # print 'max_key,d:',max_key,d
            m=0
            for i in range(int(max_key)-range_key, int(max_key)+range_key+1):
                # print 'i:',i
                if d.get(i)!=None:
                    m+=d.get(i)
            point=float(m)/length               #深度列表数据最多的总长度比值
            point1=float(length)/max_length     #深度数据的有效长度
            # print("point:%f,point1:%f"%(point,point1))
            if point>grade and point1>grade:
                return int(max_key)
            else :
                return -1 #深度数据不对  
        else :
            return -1 #深度数据不对

    def pixel2camera(self, x, y, deep):
        '''
        x   物体某点的x坐标 pixel 
        y   物体某点的y坐标 pixel
        deep    该点对应的深度数据 单位mm
        转换相机坐标系
        转换方法1
        # 0.022698268315 0.0291117414051 0.178  
        '''
        # 相机内参 
        camera_factor = 1000.0                  #深度数据单位为mm，除以1000转换为m
        camera_cx = self.intr.ppx               #图像坐标系中心点x坐标，单位mm
        camera_cy = self.intr.ppy               #图像坐标系中心点y坐标，单位mm
        camera_fx = self.intr.fx
        camera_fy = self.intr.fy
        #图像坐标系,单位mm
        Image_x= x-camera_cx
        Image_y= y-camera_cy
        #图像坐标系转换为相机坐标系.
        Zc= deep/ camera_factor                 #相机坐标系z坐标,单位m
        Xc= (Image_x/camera_fx)*Zc              #在大地坐标系放大y坐标系
        Yc= (Image_y/camera_fy)*Zc
        return (Xc,Yc,Zc)                       #返回相机坐标系下的坐标点

    def pixel2camera_api(self, x, y, deep):
        '''
        x   物体某点的x坐标 pixel 
        y   物体某点的y坐标 pixel
        deep    该点对应的深度数据 单位mm
        转换相机坐标系
        转换方法2
        [22.698266983032227, 29.696226119995117, 178.0]
        '''
        camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=self.depth_intrin, pixel=[x, y], depth=deep) #单位mm
        camera_factor = 1000.0 
        Zc= camera_coordinate[2]/ camera_factor     
        Xc= camera_coordinate[0]/ camera_factor 
        Yc= camera_coordinate[1]/ camera_factor 
        return (Xc,Yc,Zc)                       #返回相机坐标系下的坐标点

    def camera2world(self,pos):
        camera_pos = PointStamped()
        camera_pos.header.frame_id = self.camera_link     #data.header.frame_id 
        camera_pos.point.x = pos[0]
        camera_pos.point.y = pos[1]                     #camera rgb坐标转camera link坐标
        camera_pos.point.z = pos[2]
        try:
            poi = self.tf_listener.transformPoint("base_link", camera_pos)
        except:
            return -1
        return poi

    def __win_is_open(self,name):
        for i in self.open_wins:
            if i==name:
                return True
            else:
                return False
        return False

    def open_win(self,img):
        if self.__win_is_open(self.window_name)==False:    
            self.open_wins.append(self.window_name)
        cvwin.imshow(self.window_name,img)

    def close_win(self):
        if self.__win_is_open(self.window_name)==True:
            cvwin.destroyWindow(self.window_name)
            self.open_wins.remove(self.window_name)

    def object_detect(self):
        (_, rets, types, pp) = self.findObj.predict_resize(self.color_data) 
        # 寻找分值最大的
        maxval = 0
        box=[]
        type = ""
        for i in range(len(pp)):
            if pp[i] > maxval:
                maxval = pp[i]      
                box = rets[i]      
                type = types[i]
        # 使用绿色方框描述分值最大者
        if len(box)>0 :
            rect = box                         
            if rect[2]>self.MARGIN_PIX*2 and rect[3] > self.MARGIN_PIX*2:   #宽度大于两倍的边缘和长度大于两倍边缘
                rect2 = (rect[0]+self.MARGIN_PIX, rect[1]+self.MARGIN_PIX, rect[2]-self.MARGIN_PIX*2, rect[3]-self.MARGIN_PIX*2)    #rect2为处理后的方框坐标，绿色方框            
                cv2.rectangle(self.color_data, (rect2[0], rect2[1]),(rect2[0] + rect2[2], rect2[1] + rect2[3]), (0, 255, 0), 2)  #显示绿色方框
            else:
                rect2 = np.array([])        
        else:
            rect2 = np.array([])            
        self.open_win(self.color_data)
        return rect2,type       #绿色方框，左上角的坐标,和物体的宽度\高度，单位pixels

    def __locObject(self):      
      #刷新相机数据
      self.camera_data()      
      #目标检测  
      pix_pos,_=self.object_detect()                  #绿色方框，左上角的坐标,和物体的宽度\高度，单位pixels
      #深度处理
      if len(pix_pos) > 0 and pix_pos[0]-30 > 0:      #物体宽度大于30像素
          pos_center=(pix_pos[0] + pix_pos[2] / 2, pix_pos[1] + pix_pos[3] / 2 )         #中心点
          depth_center= self.Depth_data(pos_center,self.MARGIN_PIX,self.MARGIN_PIX)      #pos像素坐标(x,y)
      else:
          print("Target detection error!")
          return -1  
      if depth_center!=-1 :
          print("Center pixel:",pos_center,"center depth:",depth_center) 
      #转换相机坐标系
          camera_pos=self.pixel2camera_api(pos_center[0],pos_center[1],depth_center) 
          return camera_pos     
      else:
          print("Incomplete object depth data!")
          return -1                

    def LocObject(self,wait=20,err=0.006):                            #目标识别及稳定性确认
      stTime = time.time()
      __lastcam_pos = []
      while  len(__lastcam_pos)<5:
          __camera_pos = self.__locObject()                
          if __camera_pos == -1 :
              if wait != None and time.time()-stTime > wait:
                  return None
              continue
          stTime = time.time()
          if len(__lastcam_pos) == 0:
              __lastcam_pos = [__camera_pos]
              continue
          
          if abs(__camera_pos[0] - __lastcam_pos[-1][0]) < err and \
                  abs(__camera_pos[1] - __lastcam_pos[-1][1]) < err and \
                  abs(__camera_pos[2] - __lastcam_pos[-1][2]) < err :
              __lastcam_pos.append(__camera_pos)
          else:
              __lastcam_pos = [__camera_pos]
              
      sumx = sumy = sumz = 0
      for (Xc,Yc,Zc) in __lastcam_pos:
          sumx += Xc
          sumy += Yc
          sumz += Zc
      num=len(__lastcam_pos)
      #稳定的相机坐标系
      camera_pos=[sumx/num,sumy/num,sumz/num]    
      #相机坐标系转换为世界坐标系    
      pos=self.camera2world(camera_pos)
      return pos
      
    def stopCamera(self):
      self.pipeline.stop()

if __name__ == '__main__':
    import signal
    def quit(signum, frame):
        print('')
        print('EXIT APP') 
        sys.exit()

    signal.signal(signal.SIGINT, quit)                                
    signal.signal(signal.SIGTERM, quit)
    rospy.init_node('REALSENSE2', log_level=rospy.INFO )
    cam=RealsenseCamera()
    while True:
      pos=cam.LocObject()
      print(pos)
      time.sleep(1)

    
