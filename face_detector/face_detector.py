import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST="broker"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="facial_images"


class FaceDetector:
    def __init__(self, mqtt_client):
        self.mqtt_client = mqtt_client
    
    def process(self):
        cap = cv.VideoCapture(1)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # We don't use the color information, so might as well save space
            face_cascade = cv.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                face = gray[y:y+h, x:x+w]
                cv.imshow('frame', face)
                rc, png = cv.imencode('.png', face)
                message = png.tobytes()
                self.mqtt_client.publish(LOCAL_MQTT_TOPIC, message)

            # cv.imshow('frame', gray)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break


def on_connect(client, userdata, flags, rc):
    print("Connected facial image publisher to local broker with rc: " + str(rc))


def on_publish(client, userdata, mid):
    print("A face has just been published")



local_mqtt_client = mqtt.Client()
local_mqtt_client.on_connect = on_connect
local_mqtt_client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

face_detector = FaceDetector(local_mqtt_client)
face_detector.process()
