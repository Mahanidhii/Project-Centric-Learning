import cv2
from ultralytics import YOLO
yolo=YOLO('yolov5.pt')
videoCap=cv2.VideoCapture(0)
#function to get class colours
def getColours(cls_num):
    base_colors=[(255,0,0),(0,255,0),(0,0,255)]
    color_index=cls_num % len(base_colors)
    increments=[(1,-2,1),(-2,1,-1),(1,-1,2)]
    color=[base_colors[color_index][i]+increments[color_index][i] * 
    (cls_num//len(base_colors)) % 256 for i in range(3)]
    return tuple(color)
while True:
    ret,frame=videoCap.read()
    if not ret:
        continue
    results=yolo.track(frame,stream=True)
    for result in results:
        classes_names=result.names #getting the class names
        for box in result.boxes:
            #checkign if confidence is greater than 40 percent
            if box.conf[0]>0.4:
                [x1,y1,x2,y2]=box.xyxy[0] #get coordinates
                x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
                cls=int(box.cls[0])#get the class
                class_name=classes_names[cls]
                colour=getColours(cls)#get the respective colours
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2) #draw the rectangle (boudning box)
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2) #put the class name and confidence on the image
    #show the image in the frame
    cv2.imshow('frame',frame)
    #ending the loop
    if cv2.waitKey(1)&0xFF==ord('q'): #Execution can be stopped by pressing 'q'
        break
videoCap.release()
cv2.destroyAllWindows() #CLoses the window when q is pressed
