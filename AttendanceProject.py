import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImageAttendance'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for i in myList:
    curImg=cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])
print(classNames)

def findEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('Attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            ds=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{ds}')

encodelistknown=findEncodings(images)
print("ENCODED THE IMAGE")

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facescurframe=face_recognition.face_locations(imgs)
    encodescurframe=face_recognition.face_encodings(imgs,facescurframe)

    for encodeface,faceLoc in zip(encodescurframe,facescurframe):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis=face_recognition.face_distance(encodelistknown,encodeface)
        #print(faceDis)
        matchindex=np.argmin(faceDis)

        if matches[matchindex]:
            name=classNames[matchindex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)