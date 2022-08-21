from operator import index
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os               #for img directry
import numpy as np



cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor=SelfiSegmentation()

fpsReader=cvzone.FPS()

listimg= os.listdir('img')
print(listimg)
imglist=[]
for imgpath in listimg:
    img=cv2.imread(f'img/{imgpath}')
    imglist.append(img)
print(len(imglist))

indeximg=0
while True:

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    blur = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)


    success, img=cap.read()
    imgOut=segmentor.removeBG(img, imglist[indeximg],threshold=0.45)   #image bg

    # colorbg = segmentor.removeBG(img,(255,0,0),threshold=0.45)        #colored bg


    # cv2.imshow("image", img)                                         #image separate
    # cv2.imshow("image Out", imgOut) 
    
    
    
    
    imgStacked=cvzone.stackImages([img,imgOut],2,1)                     #image stacked 
  
   
   
   
   
   
    _,imgStacked= fpsReader.update(imgStacked, color=(0,0,255))        # test frame rate
   
    cv2.imshow("image show",imgStacked)               #image stack
   
    print(indeximg)
    key=cv2.waitKey(1)
    if key==ord('a'):
        if indeximg>0:
            indeximg-=1
    elif key==ord('d'):
        if indeximg<len(imglist)-1:
            indeximg+=1
    elif key==ord('q'):
          break

  