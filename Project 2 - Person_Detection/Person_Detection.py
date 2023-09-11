import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0) #for webcam
#width and height
#cap.set(3,640)
#cap.set(4,480)
#cap.set(3,1280)
#cap.set(4,720)
cap = cv2.VideoCapture("../video/people.mp4")  # for videos


#making model
model = YOLO("../yolo-weight/yolov8n.pt")

classNames =["person","bicycle","car","motorbike","aeroplane","bus","train","boat","traffic light",
             "stop sign","parking meter","bench","bird","cat",
             "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
             "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
             "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
             "fork","knife","spoon","bowl","banana","apple","mango","sandwich","orange","broccoli",
             "carrot","hot dog","pizza","donut","cake","chair","sofa","potted plant","bed",
             "dining table","toilet","tv monitor","laptop","mouse","remote","keyboard","cell phone",
             "microwave","over","toaster","sink","refrigerator","book","clock","vase","scissors",
             "teddy bear","hair drier","toothbrush","notebook","watch","air phone","chair","goggles","ac","fan","bulb"]

mask = cv2.imread("mask.png")
#tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limitsUp = [103,161,296,161]
limitsDown = [527,489,735,489]

totalCountUp=[]
totalCountDown=[]

while True:
    success , img =cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(730,260))

    detections = np.empty((0,5))

    result = model(imgRegion,stream=True)
    #creating box
    for r in result:
        boxes= r.boxes
        for box in boxes:
            #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            #creating rectangles
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w,h =x2-x1,y2-y1
            #cvzone.cornerRect(img,(x1,y1,w,h),l=8)

            #finding confidense
            conf = math.ceil((box.conf[0]*100))/100

            #display confidence and class name
            #class_name
            cls = int(box.cls[0])
            currentClass=classNames[cls]
            if currentClass =="person" and conf > 0.3:
                #cvzone.putTextRect(img,f'{currentClass} ,{conf}',(max(0,x1),max(35,y1)),scale=0.8,thickness=1,
                 #              offset=3)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))


    resultsTracker=tracker.update(detections)
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result )
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}' , (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)

        cx,cy=x1+w//2 ,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsUp[0] <cx <limitsUp[2] and limitsUp[1] -20<cy<limitsUp[1] +20:
            if totalCountUp.count(id)== 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] <cx <limitsDown[2] and limitsDown[1] -20<cy<limitsDown[1] +20:
            if totalCountDown.count(id)== 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)


    #cvzone.putTextRect(img, f'Count:{len(totalCount)}',(50,50))
    cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)



    cv2.imshow("Image",img)
    #cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(1)


