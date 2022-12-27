#!/usr/bin/python3
# -*- coding: utf-8 -*-

##############################################################################################
# 文件：main.py
# 作者：Zonesion wanghao 20220412
# 说明：aiarm 主应用程序
# 修改：20221206   初始版本  
#       
# 注释：主要实现与vnode进行通讯
##############################################################################################
import rospy
import time
import copy 
import sys   
import signal
# import cvwin
import threading
import math
from geometry_msgs.msg import PoseStamped, PointStamped,Pose
from std_msgs.msg import Float32MultiArray,Int16MultiArray,Int32,Int32MultiArray
from aiarm.srv import *
import yaml
import tf
from realsense_camera import RealsenseCamera
this = sys.modules[__name__]

##############################################################################################
# 公共参数配置文件
##############################################################################################
this.config_path="/home/zonesion/catkin_ws/src/marm_controller/config/config.yaml"
with open(this.config_path, "r") as f:
    if sys.version_info < (3, 0):
        config = yaml.load(f.read())
    else:    
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

##############################################################################################
# 应用配置参数
##############################################################################################
this.g_open=config["g_open"]                    # 机械臂夹具打开角度
this.gripper_ty= True                           # 夹具极性
this.g_range=[-130,130]                         # 底层夹具范围
this.x_offset=0
this.y_offset=0
this.z_offset=0

from arm import Arm
class AiArm(Arm):
    def __init__(self,g_open):
        super(AiArm,self).__init__(g_open,gripper_ty=this.gripper_ty,arm_debug=False)        
        self.tf_listener = tf.TransformListener()
        self.cam=RealsenseCamera()

    def app():
      pos=cam.LocObject()
      print(pos)
      time.sleep(1) 
      
if __name__ == '__main__':
    def quit(signum, frame):
        print('EXIT APP') 
        sys.exit()

    signal.signal(signal.SIGINT, quit)                          
    signal.signal(signal.SIGTERM, quit)
    rospy.init_node("MARM_GRASP_NODE", log_level=rospy.INFO)         #初始化节点
    aiarm=AiArm(this.g_open)
    while True:
      time.sleep(99999)


