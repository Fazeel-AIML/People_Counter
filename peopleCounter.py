import cvzone
import cv2 as cv
from ultralytics import YOLO
import math
from sort import *

model = YOLO("yolov8n.pt")
cap = cv.VideoCapture(r"Things\people.mp4")

# Load the image you want to overlay
overlay_img = cv.imread(r"Things\graphics_people.png")  # Change "overlay_image.png" to your image file

#Loading Mask
mask = cv.imread(r"Things\mask_people.png")
mask = cv.resize(mask,(1280,720))

# Check if the image has an alpha channel. If not, add one.
if overlay_img.shape[2] == 3:  # Check if the image has 3 channels (no alpha)
    overlay_img = cv.cvtColor(overlay_img, cv.COLOR_BGR2BGRA)  # Convert to BGRA (add alpha channel)
cap.set(3,1280)
cap.set(4,720)


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalUp = []
totalDown = []

while True:
    _, frame = cap.read()
    imgmask = cv.bitwise_and(frame,mask)
    if not _:
        break
    over = cvzone.overlayPNG(frame,overlay_img,(730, 260))
    result = model(imgmask, stream=True)
    detections = np.empty((0,5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),1)
            w,h = x2-x1,y2-y1
            conf = math.ceil(int(box.conf[0])*100)/100

            cls = int(box.cls[0])
            if classNames[cls] == "person":
                # cvzone.cornerRect(frame,(x1,y1,w,h),l=9,rt=5)
                # cvzone.putTextRect(frame,f"{classNames[cls]}  {conf}",(max(0,x1),max(35,y1)),scale=0.5,thickness=1, font=cv.FONT_HERSHEY_TRIPLEX)  
                
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    resultTracker = tracker.update(detections)
    cv.line(over,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv.line(over,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5)
    
    for r in resultTracker:
        x1,y1,x2,y2,Id = r
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cvzone.cornerRect(over,(x1,y1,w,h),l=9,rt=2,colorR = (255, 0, 0))
        cvzone.putTextRect(over,f"{int(Id)}",(max(0,x1),max(35,y1)),scale=2)
        cx , cy = x1+w//2,y1+h//2
        cv.circle(over,(cx,cy),4,(255,0,255),cv.FILLED)
        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-15<cy<limitsUp[1]+15:
           if totalUp.count(Id)==0:
               totalUp.append(Id)
               cv.line(over,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,255),5)
        if limitsDown[0]<cx<limitsDown[2] and limitsDown[1]-15<cy<limitsDown[1]+15:
           if totalDown.count(Id)==0:
               totalDown.append(Id)
               cv.line(over,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,255),5)
         # # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv.putText(over,str(len(totalUp)),(929,345),cv.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv.putText(over,str(len(totalDown)),(1191,345),cv.FONT_HERSHEY_PLAIN,5,(50,50,230),7)
    cv.imshow("Vehicle Counter",over)
    cv.waitKey(1) 
cap.release()
cv.destroyAllWindows() 