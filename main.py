import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'TrainingImages'
images = []
classNames = []

myList = [f for f in os.listdir(path) if not f.startswith('.')]
# print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None: 
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(classNames)                                       #printing the list

def findEncodings(images):
    encodeList = []                                     #encoding of underlying patterns
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]        
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%m/%d/%Y,%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete!')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error! Could not open webcam!")
    exit()

while True:
    success, img = cap.read()                               #returns success capture(t/f) and the flattened image
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgs)           #HOG model: Histogram of Oriented Gradients
    encodesCurrFrame = face_recognition.face_encodings(imgs, facesCurrFrame)          #FaceNet model

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)