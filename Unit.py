import cv2
import numpy as np
import math
import imutils
import paho.mqtt.publish as publish
import socket
import pickle

UDP_IP="192.168.1.1"
UDP_PORT=5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = 0

hostname = "iot.eclipse.org" # Sandbox broker
port = 1883 # Default port for unencrypted MQTT

topic = "elec3542/test" # '/' as delimiter for sub-topics
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
def pub(mtx, pos):
    if 'A' in mtx and 'B' in mtx:
        #print("Both are there")
        publish.single(topic+"t", payload="Contested",qos=0,hostname=hostname,port=port)
    elif 'A' in mtx:
        #print("A" +str(pos) + "is There")
        publish.single(topic+"t", payload="A"+str(pos),qos=0,hostname=hostname,port=port)
    elif 'B' in mtx:
        #print("B" +str(pos)+" is there")
        publish.single(topic+"t", payload="B"+str(pos),qos=0,hostname=hostname,port=port)
#matrix = np.zeros((8,8),dtype=str)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([136,87,111])
    upper_red = np.array([180,255,255])

    lower_blue = np.array([99,115,100])
    upper_blue = np.array([160,255,255])

    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    res2 = cv2.bitwise_and(frame, frame, mask = mask2)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    fgmask = fgbg.apply(res2)

    #cv2.imshow('original', frame)

    thresh = cv2.threshold(mask2,5,255,cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations =2)
    #(_,cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    matrix = np.zeros((8,8),dtype=str)

    for c in cnts:
        if cv2.contourArea(c)< 500:
            continue
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        color = "blue"
        cv2.drawContours(frame,[c],-1,(0,255,0),2)
        cv2.circle(frame,(cX,cY),7,(255,255,255),-1)
        cv2.putText(frame, "x: {}, y: {}, team: {}".format(cX,cY,color), (cX-20,cY-20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        matrix[int(float(cY/80))][int(float(cX/80))] = "B"

    thresh2 = cv2.threshold(mask,5,255,cv2.THRESH_BINARY)[1]

    thresh2 = cv2.dilate(thresh2, None, iterations =2)
    #(_,cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh2.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        if cv2.contourArea(c)< 500:
            continue
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        color = "red"
        cv2.drawContours(frame,[c],-1,(0,255,0),2)
        cv2.circle(frame,(cX,cY),7,(255,255,255),-1)
        cv2.putText(frame, "x: {}, y: {}, team: {}".format(cX,cY,color), (cX-20,cY-20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        matrix[int(float(cY/80))][int(float(cX/80))] = "A"

    cv2.imshow('frame',frame)
    print(matrix)
    Message = pickle.dumps(matrix)
    cp1 = matrix[0:2,5:8]
    pub(cp1,1)
    cp2 = matrix[4:6,1:3]
    pub(cp2,2)
    sock.sendto(Message, (UDP_IP,UDP_PORT))
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break

cap.release()
cv2.destryAllWindows()
