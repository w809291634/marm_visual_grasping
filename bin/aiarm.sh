#!/bin/bash

sudo chmod 666 /dev/ttyXCar
sudo chmod 666 /dev/ttyUSB*
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

roslaunch aiarm aiarm-controller.launch &
sleep 15
cd /home/zonesion/catkin_ws/src/aiarm/script
kill -9 `pidof vnode-xarm`
../vnode/xarm/release/vnode-xarm &
ps -aux | grep "python3 main.py"|awk '{print $2}'|xargs kill -9
python3 main.py
sleep 99999
