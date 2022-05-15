from pydoc import classname
import jetson.inference
import jetson.utils
import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)

cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


    imgCuda = jetson.utils.cudaFromNumpy(frame)

    detections = net.Detect(imgCuda)
    for d in detections:
        print(d)
        x1,y1,x2,y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
        classname = net.GetClassDesc(d.ClassID)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
        cv2.putText(frame, classname, (x1+5, y1+15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)



    #frame = jetson.utils.cudaToNumpy(imgCuda)


    cv2.imshow('Input', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()