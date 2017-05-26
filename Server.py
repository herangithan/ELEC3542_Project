import socket
import pickle
import paho.mqtt.publish as publish
import numpy as np

m1 = np.zeros((8,8),dtype=str)
m2 = np.zeros((8,8),dtype=str)
test = np.zeros((8,8),dtype=str)
test[0][0] = "A"
test[0][1] = "B"

hostname = "iot.eclipse.org" # Sandbox broker
port = 1883 # Default port for unencrypted MQTT
topic = "elec3542/test" # '/' as delimiter for sub-topics

def pub(mtx, pos):
    if 'A' in mtx and 'B' in mtx:
        print("Both are there")
        publish.single(topic, payload="Contested",qos=0,hostname=hostname,port=port)
    elif 'A' in mtx:
        print("A" +str(pos) + "is There")
        publish.single(topic, payload="A"+str(pos),qos=0,hostname=hostname,port=port)
    elif 'B' in mtx:
        print("B" +str(pos)+" is there")
        publish.single(topic, payload="B"+str(pos),qos=0,hostname=hostname,port=port)

UDP_IP = "192.168.1.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP,UDP_PORT))

while True:
        data, addr = sock.recvfrom(1024)
        matrix = pickle.loads(data)
        if str(addr[0]) == "192.168.1.101":
                m1 = matrix
        else:
                m2 = matrix
        matrix = np.where(m1 != "",m1,m2)
        #print(matrix)
        cp1 = matrix[0:2,5:8]
        pub(cp1,1)
        cp2 = matrix[4:6,1:3]
        pub(cp2,2)
        #print(matrix)
        #print(addr[0])
        print(np.where(matrix != "",matrix,test))
