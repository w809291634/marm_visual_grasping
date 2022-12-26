#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface,Constraints,TrajectoryConstraints
import cv2                                          #opencv库
import numpy as np                                  #Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import tf                                           #tf转换库
import time
import os
import math
from threading import Lock                          #线程和事件
from threading import Lock, Event
import pyrealsense2 as rs
print("pyrealsense2 version:%s"%rs.__version__)
from actionlib import simple_action_client
#消息类型
from std_msgs.msg import String,Header,Int16,Int32
from sensor_msgs.msg import Image                   #ROS消息类型，Image
from geometry_msgs.msg import PointStamped, Pose
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene ,JointConstraint,OrientationConstraint
from trajectory_msgs.msg import JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError       #opencv和ROS之间的消息转换
from tf.msg import tfMessage
from control_msgs.msg import FollowJointTrajectoryAction   
from control_msgs.msg import FollowJointTrajectoryResult
from moveit_simple_grasps.srv import *
#模块
import threading
from obj_detection_rk3399 import detection      #插入检测模块
from src import arm_tcp_client_controller
gripper_data=Int32()
detection_target_width=30                       #单位mm 
detection_target_height=30                      #单位mm 
detection_target_length=30                      #单位mm 

def max_word(lt):
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

def stability_check(lt,tolerance=0.003):        
    '''
    lt 列表
    tolerance 公差 单位m，初始值3mm
    '''
    lt.sort()
    max=lt[len(lt)-1]
    min=lt[0]
    if abs(max-min)>tolerance :
        print(abs(max-min)) 
        return -1
    else:
        return 1

def arm_init():                                 #机械臂初始化归位
    gripper_pub=rospy.Publisher('/xcar/gripper',Int32,queue_size=1)
    moveit_commander.roscpp_initialize(sys.argv)    # 初始化move_group的API
    arm = moveit_commander.MoveGroupCommander('manipulator')    # 初始化需要使用move group控制的机械臂中的arm group
    arm.set_goal_joint_tolerance(0.001)             # 设置机械臂运动的允许误差值      
    arm.set_max_acceleration_scaling_factor(1)      # 设置允许的最大速度和加速度
    arm.set_max_velocity_scaling_factor(1)
    arm.set_named_target('home')                    # 控制机械臂先回到初始化位置
    arm.go()
    rospy.sleep(1)
    gripper_data.data=70
    gripper_pub.publish(gripper_data)              
    rospy.loginfo("Open the fixture")
    return arm,gripper_pub

def arm_LocObject(num_numbers=3):                            #目标识别及稳定性确认
    locobj = LocObject(detection.pilldetect)        #定义一个目标识别类
    locobj.camera_init(424,240)
    # locobj.camera_init(640,360)
    sumx = sumy = sumz = num =0
    lt_x = []
    lt_y = []
    lt_z = []
    while num<num_numbers:
        ret = locobj.locObject()                    #相机坐标系
        if ret != -1 :
            if num==0:
                ret1=ret
            if abs(ret1[0]-ret[0])>0.003 or abs(ret1[1]-ret[1])>0.003 or abs(ret1[2]-ret[2])>0.003:
                sumx = sumy = sumz = num =0
                lt_x = []
                lt_y = []
                lt_z = []
                rospy.logwarn("The target is moving! Restart!")
            lt_x.append(ret[0])
            lt_y.append(ret[1])
            lt_z.append(ret[2])
            ret1=ret
            sumx+=ret[0]
            sumy+=ret[1]
            sumz+=ret[2]
            num+=1
            if num==num_numbers :
                if stability_check(lt_x)!=1 or  stability_check(lt_y)!=1 or stability_check(lt_z)!=1:
                    sumx = sumy = sumz = num =0
                    lt_x = []
                    lt_y = []
                    lt_z = []
                    rospy.logwarn("The target is moving! Restart!") 
    camera_pos=(sumx/num,sumy/num,sumz/num)         #稳定的相机坐标系下的目标
    world_pos=locobj.camera_to_world_coord(camera_pos)  #相机坐标系转换为世界坐标系
    if world_pos!=-1:
        Object_pose=Pose()
        Object_pose.position.x=world_pos.point.x
        Object_pose.position.y=world_pos.point.y
        Object_pose.position.z=world_pos.point.z
        Object_pose.orientation.x=0
        Object_pose.orientation.y=0
        Object_pose.orientation.z=0
        Object_pose.orientation.w=1
        locobj.__del__()
        return Object_pose                          #返回目标位姿和类
    else:
        rospy.logerr("Target coordinate system error")

def moveto(pos):                                #机械臂移动
    rospy.loginfo("The mechanical arm begins to move") 
    # 控制机械臂先回到初始化位置
    arm.set_named_target('home')
    arm.go()
    rospy.sleep(0.5)
    # 移动到预抓取位
    joint_positions = pos.ik_solutions[1].positions
    arm.set_joint_value_target(joint_positions)
    arm.go()
    rospy.sleep(0.5)
    # 移动到抓取位
    joint_positions = pos.ik_solutions[0].positions
    arm.set_joint_value_target(joint_positions)       
    arm.go()
    rospy.sleep(1)
    global gripper_data
    gripper_data.data=10
    gripper_pub.publish(gripper_data)  
    rospy.loginfo( "Close the fixture")              
    rospy.sleep(1)  
    # 移动到预抓取位
    joint_positions = pos.ik_solutions[1].positions
    arm.set_joint_value_target(joint_positions)
    arm.go()
    rospy.sleep(0.5)
    # 移动到过渡位
    joint_positions = [joint_positions[0], 0.5,joint_positions[2], joint_positions[3], joint_positions[4]]
    arm.set_joint_value_target(joint_positions)
    arm.go()
    rospy.sleep(0.5)
    # 控制机械臂先回到初始化位置
    arm.set_named_target('home')
    arm.go()
    rospy.sleep(1)
    # 关闭并退出moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)

class LocObject:
    '''目标识别，并获取目标三维坐标'''
    def __init__(self, findObjCall, onObjCall=None, MARGIN_PIX=7):      #MARGIN_PIX为绿色边缘和红色边缘距离
        self.onRSObject = onObjCall
        self.findObj = findObjCall          #findObj为OPENCV目标检测函数
        self.MARGIN_PIX = MARGIN_PIX        #在原检测框里margin

        self.__push = True                  #处理结束标志
        self.__window = False               #视频窗口打开
        self.__intr_init= False             #相机内参获取完成
        self.color_data=Image()
        self.dep_data=Image()

        self.__evt = Event()                #实例化一个Event类
        self.__lock= Lock()                 #实例化一个Lock类

    def camera_init(self,WIDTH,HEIGHT):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)     #使能深度相机
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)    #使能彩色相机
        self.profile = self.pipeline.start(config)
        self.__camera_enable=True
        # 保存相机内参
        frames = self.pipeline.wait_for_frames()            #等待相机坐标系生成
        color_frame = frames.get_color_frame()         #获取彩色相机坐标系
        self.intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {'fx': self.intr.fx, 'fy': self.intr.fy,
                            'ppx': self.intr.ppx, 'ppy': self.intr.ppy,
                            'height': self.intr.height, 'width': self.intr.width,
                            'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                            }
        rospy.logwarn(camera_parameters)
        # 保存深度参数
        align_to = rs.stream.color                              #统一对齐到彩色相机
        align = rs.align(align_to)
        aligned_frames = align.process(frames)                  #对齐后的相机坐标系
        aligned_depth_frame = aligned_frames.get_depth_frame()  #对齐到彩色相机后的深度相机坐标系
        self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        rospy.logwarn(self.depth_intrin)

    def camera_data(self):
        if self.__camera_enable is True:  
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
        else:
            rospy.logerr("Camera is not enable!")
            self.__intr_init= False             #相机内参获取完成
            self.__camera_enable = False        #相机没有使能

    def Invalid_Depth_Band(object_distance):                #无效深度带
        invalid_Depth_Band=55*camera_fx/object_distance     #单位像素pixels
        print(invalid_Depth_Band)

    def object_detect(self):
        _, box, type = self.findObj(self.color_data)   #_:处理后的图像，带有红色方框.与cv_image同一内存    #box:  返回左上角的坐标,和物体的宽度\高度，红色方框 #type:标签
        if len(box)>0 :
            rect = box                          #取box数据，返回左上角的坐标,和物体的宽度\高度，
            if rect[2]>self.MARGIN_PIX*2 and rect[3] > self.MARGIN_PIX*2:   #宽度大于两倍的边缘和长度大于两倍边缘
                rect2 = (rect[0]+self.MARGIN_PIX, rect[1]+self.MARGIN_PIX, rect[2]-self.MARGIN_PIX*2, rect[3]-self.MARGIN_PIX*2)    #rect2为处理后的方框坐标，绿色方框            
                cv2.rectangle(self.color_data, (rect2[0], rect2[1]),(rect2[0] + rect2[2], rect2[1] + rect2[3]), (0, 255, 0), 2)  #显示绿色方框
            else:
                rect2 = np.array([])        
        else:
            rect2 = np.array([])            
        self.__window = True                    #图像显示窗口打开
        cv2.imshow("Image window", self.color_data)    #打开图像显示
        cv2.waitKey(1)  
        return rect2,type   #绿色方框，左上角的坐标,和物体的宽度\高度，单位pixels
            
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
            max_key,d=max_word(list_deep)
            # print 'max_key,d:',max_key,d
            m=0
            for i in range(int(max_key)-range_key, int(max_key)+range_key+1):
                # print 'i:',i
                if d.get(i)!=None:
                    m+=d.get(i)
            point=float(m)/length               #深度列表数据最多的总长度比值
            point1=float(length)/max_length     #深度数据的有效长度
            # print 'point:',point,'point1:',point1
            if point>grade and point1>grade:
                return int(max_key)
            else :
                return -1 #深度数据不对  
        else :
            return -1 #深度数据不对

    def camera_coord_pos(self, x, y, deep):
        '''
        x   物体某点的x坐标 pixel 
        y   物体某点的y坐标 pixel
        deep    该点对应的深度数据 单位mm
        转换相机坐标系
        转换方法1
        # 0.022698268315 0.0291117414051 0.178  
        '''
        # 相机内参 通过命令获取
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

    def camera_coord_pos_api(self, x, y, deep):
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

    def camera_to_world_coord(self,pos):
        camera_pos = PointStamped()
        camera_pos.header.frame_id = "camera_link"     #data.header.frame_id 
        camera_pos.point.x = pos[0]
        camera_pos.point.y = pos[1]                     #camera rgb坐标转camera link坐标
        camera_pos.point.z = pos[2]
        try:
            tf_listener = tf.TransformListener()
            point_trans = tf_listener.waitForTransform("base_link", camera_pos.header.frame_id, rospy.Time(0), rospy.Duration(1))
            poi = tf_listener.transformPoint("base_link", camera_pos)
        except:
            return -1
        return poi



    def locObject(self, wait=None): #等待目标识别，并获取目标三维坐标，wait为等待超时时间
        while True:
            if self.__camera_enable is True: 
                self.camera_data()
            else:
                rospy.logerr("Camera is not enable!")
                return -1
            #目标检测  
            pix_pos,_=self.object_detect()                  #绿色方框,左上角的坐标,和物体的宽度\高度，单位pixels
            #深度处理
            if len(pix_pos) > 0 and pix_pos[0]-30 > 0:      #物体宽度大于30像素
                pos_center = (pix_pos[0] + pix_pos[2] / 2, pix_pos[1] + pix_pos[3] / 2 )        #中心点
                depth_center = self.Depth_data(pos_center,self.MARGIN_PIX,self.MARGIN_PIX)      #pos像素坐标(x,y)
                # pos=(pix_pos[0]+self.MARGIN_PIX*2 , pix_pos[1]+self.MARGIN_PIX*2 )              #左上角
                # depth_diagonal_1 = self.Depth_data(pos,self.MARGIN_PIX,self.MARGIN_PIX)         #pos像素坐标(x,y)
                # pos=(pix_pos[0]+pix_pos[2]-self.MARGIN_PIX*2 , pix_pos[1]+pix_pos[3]-self.MARGIN_PIX*2 )      #右下角
                # depth_diagonal_2 = self.Depth_data(pos,self.MARGIN_PIX,self.MARGIN_PIX)         #pos像素坐标(x,y)
            else:
                rospy.logwarn("Target detection error!")
                continue
            # if depth_center!=-1 and depth_diagonal_1!=-1 and depth_diagonal_2!=-1 :
            if depth_center!=-1:
                # print "Center pixel:",pos_center,"center depth:",depth_center
                print("Center pixel:%s center depth:%s"%(pos_center,depth_center))

            #转换相机坐标系
                global detection_target_width
                depth_center=depth_center+detection_target_width/2                      #偏移到物体中心
                camera_pos=self.camera_coord_pos_api(pos_center[0],pos_center[1],depth_center) 
                return camera_pos     
            else:
                rospy.logwarn("Incomplete object depth data!")
                continue               


    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('obj_detect', anonymous=True)   #设置节点名称
    
    import sys                                                                  
    import signal
    
    def quit(signum, frame):
        print('') 
        print('stop fusion') 
        sys.exit()
        
    signal.signal(signal.SIGINT, quit)                                
    signal.signal(signal.SIGTERM, quit)

    arm,gripper_pub = arm_init()
    Object_pose = arm_LocObject(3)
    print("Object_pose:",Object_pose)
    if Object_pose != -1:
        rospy.wait_for_service("/grasp_filter_test/GenerateSolutions",timeout=10)  
        rospy.loginfo('connecting to the server successfully')
        try:
            Solutions_client = rospy.ServiceProxy('/grasp_filter_test/GenerateSolutions', GenerateSolutions)  
            response = Solutions_client(Object_pose)      
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)
        if response.ik_solutions[0].positions: 
            rospy.loginfo("Query pose completed!")
            moveto(response)                             #移动到轨迹点
        else:
            rospy.logerr("Query pose fail!")
    else:
        rospy.logerr("Target coordinate system error")


    