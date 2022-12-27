#!/usr/bin/python3
# -*- coding: utf-8 -*-

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
this.g_open=config["g_open"]          # 机械臂夹具打开角度
this.gripper_ty= True                 # 夹具极性
this.g_range=[-130,130]               # 底层夹具范围
#视觉抓取修正参数
this.x_offset=0                       # 相机坐标系的平移修正
this.y_offset=0           
this.z_offset=0
this.x_factor=1.2                     # x轴上中心点两侧放大
this.err=0.003                        # 稳定性检查

from arm import Arm
class AiArm(Arm):
    def __init__(self,g_open):
        super(AiArm,self).__init__(g_open,gripper_ty=this.gripper_ty,arm_debug=False)        
        self.tf_listener = tf.TransformListener()
        self.cam=RealsenseCamera()

def armAPP():
    arm=AiArm(this.g_open)
    arm.all_gohome()  
    while not rospy.is_shutdown(): 
      pos=arm.cam.LocObject(err=this.err,x_factor=this.x_factor)    #目标定位
      if pos!=None:
        Object_pose=Pose()
        Object_pose.position.x=pos.point.x +this.x_offset     
        Object_pose.position.y=pos.point.y +this.y_offset         
        Object_pose.position.z=pos.point.z +this.z_offset     
        Object_pose.orientation.x=0
        Object_pose.orientation.y=0
        Object_pose.orientation.z=0
        Object_pose.orientation.w=1
        response = arm.Solutions_client(Object_pose)                #向服务器查询机械臂最佳的抓取姿态
        arm.cam.close_win()
        if  len(response.ik_solutions[0].positions)>0:
          arm.arm_goHome()
          rospy.sleep(0.1)
          # 移动到预抓取位
          joint_positions = response.ik_solutions[1].positions
          arm.set_joint_value_target(joint_positions)
          rospy.sleep(0.1)
          # 移动到抓取位
          joint_positions = response.ik_solutions[0].positions
          arm.set_joint_value_target(joint_positions)       
          rospy.sleep(0.1)
          # arm.setGripper(True)
          rospy.sleep(0.1)
          # 移动到预抓取位
          joint_positions = response.ik_solutions[1].positions
          arm.set_joint_value_target(joint_positions)
          rospy.sleep(0.1)
          arm.arm_goHome()
          rospy.sleep(1)
          # arm.setGripper(False)
      time.sleep(1)

if __name__ == '__main__':
    def quit(signum, frame):
        print('EXIT APP') 
        sys.exit()

    signal.signal(signal.SIGINT, quit)                          
    signal.signal(signal.SIGTERM, quit)
    rospy.init_node("MARM_GRASP_NODE", log_level=rospy.INFO)         #初始化节点
    armAPP()


