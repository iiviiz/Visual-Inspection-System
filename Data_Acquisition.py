##----------------------------------------------------
## Created by Ivan Luiz de Oliveira
## Unesp Sorocaba Dez. 2018
##----------------------------------------------------

import cv2

##------------------------------------------------------------
## Read config file.
##------------------------------------------------------------
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
imgSize = int(config_data[6][1])
classification_header = config_data[11][1]

##------------------------------------------------------------
## Test and configurate camera image.
##------------------------------------------------------------
part_quality=[0,0,0,0]
partCenter=[x_partCenter,y_partCenter]
sotage_imageSize = [imgSize , imgSize]
testCamera=[False,]
input("Press any key to begin...")
while(testCamera[0]==False):
        capture = cv2.VideoCapture(cameraPort)
        testCamera=capture.read()
        if(testCamera[0]==False):        
            input('Camera not found!\nPress any key to try again...')
capture.set(3,capture_width)#640
capture.set(4,capture_height)#480

##------------------------------------------------------------
## Read or create classification file
##------------------------------------------------------------
try:
    f=open('classification_data.dat','r')
    list_lines=f.readlines()
    f.close()
    if len(list_lines) == 1:
        image_number = 1
    else:
        temp = list_lines[-1].split(',')
        temp = temp[0].split('_')
        image_number = int(temp[1])
        image_number += 1
except IOError:
    f=open('classification_data.dat','w')
    f.write(classification_header)
    f.close()
    image_number = 1

##------------------------------------------------------------
## Take picture, ask for classification inputs and store both
##------------------------------------------------------------
print('\nImage size:',sotage_imageSize)
keep_runing=''
while keep_runing != 'n':
    allowSave = False
    while allowSave == False:
        for i in range(2): ret, frame = capture.read()
        frame=frame[int(partCenter[0]-(sotage_imageSize[0]/2)):int(partCenter[0]+(sotage_imageSize[0]/2)),int(partCenter[1]-(sotage_imageSize[1]/2)):int(partCenter[1]+(sotage_imageSize[1]/2))]
        cv2.imshow('Preview Image! (Save?[Y/N])',frame)
        #print('Preview the image!')
        waitInput = cv2.waitKey(0)
        allowSave =((waitInput& 0xFF == ord('y')) or((waitInput& 0xFF == ord('Y'))))
    cv2.destroyAllWindows()
    image_name = 'image_'+str(image_number)
    part_color = input("Inform the part's color [B=black / R=red / S=silver]:\t")
    while(part_color.lower()!='b' and part_color.lower()!='r' and part_color.lower()!='s' ):
        part_color = input("Invalid input, try again:\t")
    done_classifying='n'
    while(done_classifying.lower()=='n'):
        quality_input = input("Classify the part quality:\n[G=Good / H=no Hole / P=have Pin / T=broken Tooth:\t")
        while(quality_input.lower()!='g' and quality_input.lower()!='h' and quality_input.lower()!='p' and quality_input.lower()!='t'):
            quality_input = input("Invalid input, try again:\t")    
        if quality_input.lower() == 'g': part_quality[0] = 1
        elif quality_input.lower() == 'h': part_quality[1] = 1
        elif quality_input.lower() == 'p': part_quality[2]= 1
        elif quality_input.lower() == 't': part_quality[3]= 1
        done_classifying=input("Done classifying? [Y/N]?\t")
    print("Appended to .dat: ",end='')
    print(image_name,part_color,part_quality,sep=',')
    cv2.imwrite('Database/'+image_name+'.jpeg',frame)
    f=open('classification_data.dat','a+')
    f.write('\n{},{},{},{},{},{}'.format(image_name,part_color.lower(),part_quality[0],part_quality[1],part_quality[2],part_quality[3]))
    f.close()
    if(image_number%5==0):
        print('\nChange part COLOR!')
    image_number += 1
    keep_runing = input("Keep runing ? [Y/N]?\t")
capture.release()
print("-----------------------------\nEnd.")