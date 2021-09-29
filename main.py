import cv2
import face_recognition as fr
import numpy as np
cr7 = fr.load_image_file("cr7.jpg")
cr7 = cv2.cvtColor(cr7,cv2.COLOR_BGR2RGB)
loc = fr.face_locations(cr7)[0]
enco = fr.face_encodings(cr7)[0]
cr7 = cv2.rectangle(cr7,(loc[3],loc[0]),(loc[1],loc[2]),(255,0,0),2)

# for test image
cr7t = fr.load_image_file("cr7t.jpg")
cr7t = cv2.cvtColor(cr7t,cv2.COLOR_BGR2RGB)
encot = fr.face_encodings(cr7t)[0]

result = fr.compare_faces([enco],encot)
rtp = fr.face_distance([enco],encot)
cr7t = cv2.putText(cr7t,f'{result[0]} {(1-round(rtp[0],2))}',(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
print(result,(1-rtp)*100)
cv2.imshow("img",cr7)
cv2.imshow("img2",cr7t)
cv2.waitKey(0)