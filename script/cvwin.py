#!/usr/bin/python
# -*- coding: utf-8 -*-
import threading
import cv2
import sys
import time
import copy
from collections import OrderedDict

this = sys.modules[__name__]
this.showwins={}            #{"name":{"img":img,"update":True},}
this.destroywins=[]
this.resizewins=[]          #[  [] ]
this.namedWindows={}        #{"name":True}
this.createTrackbars=OrderedDict()     #{"winname":{"trackbarname":{"param":[trackbarname,winname,value,count,TrackbarCallback],"update":True,"value":0},},}
this.createTrackbars_win=OrderedDict() #{"winname":createTrackbars_win}
this.Trackbarvalues={}      #{"winname":{trackbarname:value},}

def namedWindow(name):
    this.namedWindows[name]=True        #ture：需要启动

def createTrackbar(trackbarname,winname,value,count,TrackbarCallback):
    Trackbar={
        "param":[trackbarname,winname,value,count,TrackbarCallback],
        "update":True,
        "value":value
    }
    this.createTrackbars_win[trackbarname]=Trackbar
    this.createTrackbars[winname]=copy.deepcopy(this.createTrackbars_win)

def getTrackbarPos(trackbarname, winname):
    return this.createTrackbars[winname][trackbarname]["value"]

def imshow(name,img):
    win={
        "img":img,
        "update":True
    }
    this.showwins[name]=win

def destroyWindow(name):
    this.destroywins.append(name)

def resizeWindow(name,x,y):
    this.resizewins.append([name,x,y])

this.st=time.time()
def cvwin():
    while True:
        for i in this.namedWindows.keys():
            if this.namedWindows[i]==True:
                cv2.namedWindow(i)
                # cv2.namedWindow(i,cv2.WINDOW_NORMAL)
                this.namedWindows[i]=False
        
        for m in this.createTrackbars.keys():   #win
            if this.namedWindows[m]==False:    #check win is open
                for n in this.createTrackbars[m].keys(): #Trackbar
                    try :
                        t=cv2.getWindowProperty(m,cv2.WND_PROP_VISIBLE)
                    except:
                        pass
                    if this.createTrackbars[m][n]["update"]==True and t and this.namedWindows[m]==False:   #检查窗口打开:
                        data=this.createTrackbars[m][n]["param"]
                        cv2.createTrackbar(data[0],data[1],data[2],data[3],data[4])
                        this.createTrackbars[m][n]["update"]=False

        if len(this.resizewins)>0:
            rswin=this.resizewins[0]
            try :
                if cv2.getWindowProperty(rswin[0],cv2.WND_PROP_VISIBLE):
                    del this.resizewins[0]
                    cv2.resizeWindow(rswin[0],rswin[1],rswin[2])
            except:
                pass
            
        for i in this.showwins.keys():
            win=this.showwins[i]
            if win["update"]==True:
                cv2.imshow(i, win["img"])
                win["update"]=False

        if len(this.destroywins)>0:
            try:
                delwin=this.destroywins[0]
                if cv2.getWindowProperty(delwin,cv2.WND_PROP_VISIBLE):
                    del this.destroywins[0]
                    del this.showwins[delwin]
                    cv2.destroyWindow(delwin)
            except Exception as e:
                print(e)
                print("destroyWindow error! retry")
                pass  
                
        for m in this.createTrackbars.keys():   #win
            try :
                if cv2.getWindowProperty(m,cv2.WND_PROP_VISIBLE):   #检查窗口打开
                    for n in this.createTrackbars[m].keys():
                        if this.createTrackbars[m][n]["param"]:
                            data=this.createTrackbars[m][n]["param"]
                            this.createTrackbars[m][n]["value"]=cv2.getTrackbarPos(data[0],data[1])
                            # print(m,n,this.createTrackbars[m][n]["value"])
            except:
                break
        if time.time()-this.st>30:
            print("Win Thread Running")
            this.st=time.time()
        cv2.waitKey(10)

t = threading.Thread(target=cvwin)
t.setDaemon(True)
t.start()


