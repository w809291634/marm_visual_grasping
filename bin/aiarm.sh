#!/bin/bash

sudo chmod 666 /dev/ttyXCar
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

roslaunch marm_visual_grasping aiarm-controller.launch &
sleep 15
cd /home/zonesion/catkin_ws/src/marm_visual_grasping/script
ps -aux | grep "python3 main.py"|awk '{print $2}'|xargs kill -9
python3 main.py
sleep 99999
