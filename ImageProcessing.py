import cv2
import ObjectDetectionMNV2 as ObjDet
import os


cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)


if not cap.isOpened():
    raise IOError("Cannot open webcam")

myModel = ObjDet.mnSSD("ssd-mobilenet-v2", 0.5)

count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    objects = myModel.detect(frame, True)

    if len(objects)!=0:
        if max(objects) > ((640*360)/26):
            count+=1
        else:
            count-=1

        if count > 48 :

            dur = "30"
            bashcmd = "timeout "+ dur + " gst-launch-1.0 udpsrc do-timestamp=true port= 5600 caps='application/x-rtp' ! rtph264depay ! h264parse disable-passthrough=true ! avdec_h264 ! videoflip method=rotate-180 ! xvimagesink sync=false"
            os.system(bashcmd)
            count = 5
            
        elif count < 0:
            count = 0
            print("durdu")
        

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
