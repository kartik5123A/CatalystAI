from ultralytics import YOLO
import cv2
import pandas as pd
from tracker import *
 
cap = cv2.VideoCapture("D:/Catalyst Python/yolo_practice/videos/highway_mini.mp4")
 
model = YOLO("D:/Catalyst Python/yolo_practice/yolov10x.pt")
 
classNames = model.names

tracker = Tracker()
count = 0

down = {}
 
while True:
    success, img = cap.read()
    if not success:
        break
    count += 1
    frame = cv2.resize(img, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        id = int(row[5])
        class_name = classNames[id]
        if "car" in class_name:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4)//2
        cy = int(y3 + y4)//2

        # cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
        # cv2.rectangle(frame, (x3, y3), (x4, y4), (0,0,255), 2)
        
        red_line_y = 198
        blue_line_y = 268
        offset = 7

        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = cy

            if id in down:
                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)

    text_color = (255, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)

    cv2.line(frame, (172, 198), (774, 198), red_color, 3)
    cv2.putText(frame, ('red line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (8, 268), (927, 268), blue_color, 3)
    cv2.putText(frame, ('blue line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.imshow("frames", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break
 
cap.release()
cv2.destroyAllWindows()