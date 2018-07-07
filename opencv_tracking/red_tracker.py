
# coding: utf-8

# In[1]:

from collections import deque

import numpy as np
import cv2
import imutils
import time


# In[ ]:

red_lower=(0,50,50)
red_upper=(5,255,255)

pts=deque(maxlen=64)
vs=cv2.VideoCapture(0)

#time.sleep(2.0)
count=0

# In[ ]:
ret,frame=vs.read()
while ret:
    ret,frame=vs.read()
    frame=imutils.resize(frame,width=600)
    blurred=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,red_lower,red_upper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if imutils.is_cv2() else cnts[1]
    center=None
    if len(cnts)>0:
    	#print("red found")

    	c=max(cnts,key=cv2.contourArea)
    	((x,y),radius)=cv2.minEnclosingCircle(c)
    	#print(radius)
    	M=cv2.moments(c)
    	center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
    	if radius>20:
    		cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,0),4)
    		cv2.circle(frame,center,5,(0,255,255),-1)
    pts.appendleft(center)
    img=np.zeros((512,512,3),np.uint8)
    for i in range(1,len(pts)):
    	if pts[i-1] is None or pts[i] is None:
    		continue
    				
    	cv2.line(img,pts[i-1],pts[i],(0,255,255),4)	
    	#output=img.copy()
    			
    	cv2.imshow('Video Input',frame)
    	cv2.imshow('Tracking path',img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circle=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,100)
    if circle is not None:
    	print('circle found')

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    else:
        pass      


# In[ ]:



