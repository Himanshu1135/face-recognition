import cv2
import face_recognition as fr
import numpy as np
import os

path = 'faces'
mylist = os.listdir(path)
images = []
names = []
print(mylist)
for i in mylist:
    cimg = cv2.imread(f'{path}/{i}')
    images.append(cimg)
    names.append(os.path.splitext(i)[0])
# print(images)

def findingfaces(a):
    encolist = []
    for j in images:
        j = cv2.cvtColor(j,cv2.COLOR_BGR2RGB)
        enco = fr.face_encodings(j)[0]
        encolist.append(enco)
    return (encolist)
encolist = findingfaces(images)
print(len(encolist))

cap = cv2.VideoCapture(0)
while True:
    sucess,frames = cap.read()
    frame = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)

    frameloc = fr.face_locations(frame)
    frameencode = fr.face_encodings(frame,frameloc)

    for fl,fe in zip(frameloc,frameencode):
        match = fr.compare_faces(encolist,fe)
        matchdis = fr.face_distance(encolist,fe)
        print(matchdis)

        matchid = np.argmin(matchdis)
        if match[matchid]:
            name =names[matchid]
            print(name)
            frame = cv2.putText(frames,name,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)


    cv2.imshow("frames",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break






