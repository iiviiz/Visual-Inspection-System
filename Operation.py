##----------------------------------------------------
## Created by Ivan Luiz de Oliveira
## Unesp Sorocaba Dez. 2018
##----------------------------------------------------



##---------------------------------------------------------
##---------------------------------------------------------
## THIS CODE ONLY WORKS ON THE ST-ONE BOARD (RASPBERRY PI)
##---------------------------------------------------------
##---------------------------------------------------------



from keras.models import load_model
import numpy as np
import cv2
import RPi.GPIO as GPIO
from time import sleep

##----------------------------------------------------
## Read config file.
##----------------------------------------------------
config=open('config.cfg','r')
config_data=config.read()
config.close()
config_data=config_data.split('::')
config_data=config_data[1].split('\n')
for i in range(1,len(config_data)):
    config_data[i] = config_data[i].split('==')
cameraPort = int(config_data[1][1])
capture_width = int(config_data[2][1])
capture_height = int(config_data[3][1])
x_partCenter = int(config_data[4][1])
y_partCenter = int(config_data[5][1])
storage_imgSize = int(config_data[6][1])
in_address = config_data[7][1]
out_address = config_data[8][1].split(']')
out_address = out_address[0].split('[')
out_address = out_address[1].split(',')
enable_collor = int(config_data[9][1])
enable_quality = int(config_data[10][1])
training_imgsize = int(config_data[12][1])
enable_preview = int(config_data[13][1])
enable_startup_run = int(config_data[14][1])

partCenter = [x_partCenter,y_partCenter]
storage_imageSize = [storage_imgSize , storage_imgSize]
training_imagesize = [training_imgsize,training_imgsize]
testCamera = [False,]

##---------------------------------------------------------
## Setup ST-One IOs.
##---------------------------------------------------------
IO_mapping = {"OUT1":13,"OUT2":12,"OUT3":25,"OUT4":24,"IN1":19,"IN2":6,"IN3":5,"IN4":22}
in_address=IO_mapping[in_address]
out_address[0]=IO_mapping[out_address[0]]
out_address[1]=IO_mapping[out_address[1]]

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(in_address, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
GPIO.setup(out_address[0], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(out_address[1], GPIO.OUT, initial=GPIO.LOW)

##---------------------------------------------------------
## Setup operation mode and test camera
##---------------------------------------------------------

if enable_startup_run==1:
    enable_preview=0

if enable_collor==1:
        model_collor = load_model('models/color/color_ANN_model.h5')
if enable_quality==1:
        model_quality = load_model('models/quality/quality_CNN_model.h5')
        
while(testCamera[0]==False):
        capture = cv2.VideoCapture(cameraPort)
        testCamera=capture.read()
        if(testCamera[0]==False):        
            input('Camera not found!\nPress any key to try again...')
capture.set(3,capture_width)#640
capture.set(4,capture_height)#480

##---------------------------------------------------------
## Operation loop
##---------------------------------------------------------
print("Program Running...")
while True:
    camera_trigger = GPIO.input(in_address)
    while camera_trigger==0:
        sleep(0.1)
        camera_trigger=GPIO.input(in_address)
    print("trigger!")
    ret, frame = capture.read()
    for i in range(10): ret, frame = capture.read()
    current_image=frame[int(partCenter[0]-(storage_imageSize[0]/2)):int(partCenter[0]+(storage_imageSize[0]/2)),int(partCenter[1]-(storage_imageSize[1]/2)):int(partCenter[1]+(storage_imageSize[1]/2))]
    current_image = cv2.resize(current_image, (training_imagesize[0], training_imagesize[1]))
    if enable_preview==1:
        cv2.imshow('Preview Image!',current_image)
        print('Preview the image!')
        waitInput = cv2.waitKey(0)
        cv2.destroyAllWindows()
    current_image = np.array(current_image) / 255.0
    current_image = np.expand_dims(current_image,aaxis=0)
    
    if enable_collor==1:
        collor_predict = model_collor.predict_classes(current_image)
        print(collor_predict)
        if collor_predict[0]==0:
            print("Part color: BLACK")
            GPIO.output(out_address[0], GPIO.HIGH)
            GPIO.output(out_address[1], GPIO.HIGH)
        elif collor_predict[0]==1:
            print("Part color: RED")
            GPIO.output(out_address[0], GPIO.LOW)
            GPIO.output(out_address[1], GPIO.HIGH)
        elif collor_predict[0]==2:
            print("Part color: SILVER")
            GPIO.output(out_address[0], GPIO.HIGH)
            GPIO.output(out_address[1], GPIO.LOW)
    
    if enable_quality==1:
        quality_predict = model_quality.predict_classes(current_image)
        print(quality_predict)
        if quality_predict[0]==0:
            GPIO.output(out_address[0], GPIO.LOW)
            GPIO.output(out_address[1], GPIO.HIGH)
        elif quality_predict[0]==1:
            GPIO.output(out_address[0], GPIO.HIGH)
            GPIO.output(out_address[1], GPIO.LOW)

            
    print("pallet liberado!")
    while camera_trigger==1:
        sleep(0.1)
        camera_trigger=GPIO.input(in_address)
    GPIO.output(out_address[0], GPIO.LOW)
    GPIO.output(out_address[1], GPIO.LOW)
    print('pallet deixou a estacao!')
        
capture.release()
GPIO.cleanup()
print("-----------------------------\nEnd.")
