from pydoc import classname
import jetson.inference
import jetson.utils
import cv2

class mnSSD():
    def __init__(self, path, threshold):
        self.path = path
        self.threshold = threshold
        self.net = jetson.inference.detectNet(self.path, self.threshold)

    def detect(self, frame, display = False):
        imgCuda = jetson.utils.cudaFromNumpy(frame)
        detections = self.net.Detect(imgCuda, overlay = "OVERLAY_NONE")

        objects = []
        for d in detections:
            className = self.net.GetClassDesc(d.ClassID)
            objects.append([className, d])

            if display:
                x1,y1,x2,y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
                cv2.putText(frame, className, (x1+5, y1+15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
                cv2.putText(frame, f'FPS: {int(self.net.GetNetworkFPS())}', (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)


        return objects


def main():

    cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    myModel = mnSSD("ssd-mobilenet-v2", 0.5)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        objects = myModel.detect(frame, True)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        

if __name__ == "__main__":
    main()