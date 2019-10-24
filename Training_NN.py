##----------------------------------------------------
## Created by Ivan Luiz de Oliveira
## Unesp Sorocaba Dez. 2018
##----------------------------------------------------

# import the necessary packages
import datetime
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation
from keras.layers.core import Flatten
#from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import random
#import pickle
import cv2

##----------------------------------------------------
## Read config file.
##----------------------------------------------------
config=open('config.cfg','r')
config_data=config.read()
config.close()
config_data=config_data.split('::')
config_data=config_data[2].split('\n')
for i in range(1,len(config_data)):
    config_data[i] = config_data[i].split('==')
train_per = float(config_data[1][1])
validat_per = float(config_data[2][1])
test_per =float(config_data[3][1])
image_color_type = int(config_data[4][1])
number_filters = config_data[6][1].split(']')
number_filters = number_filters[0].split('[')
number_filters = number_filters[1].split(',')
filter_size = int(config_data[7][1])
number_dense_quality = int(config_data[8][1])
number_dense_color = int(config_data[9][1])
dropout_quality = float(config_data[10][1])
lr_inicial_color = float(config_data[11][1])
lr_inicial_quality = float(config_data[12][1])
Epochs_quality = int(config_data[13][1])
Epochs_color = int(config_data[14][1])
color_earlystop_patience = int(config_data[15][1])
quality_earlystop_patience = int(config_data[16][1])
Bath_size = int(config_data[17][1])
random_seed = int(config_data[18][1])
image_extension = config_data[19][1]
img_size = int(config_data[23][1])
image_size = [img_size,img_size]
random.seed(random_seed)

##----------------------------------------------------
## Load Images and classification data
##----------------------------------------------------
print("------------------------------------------------------")
print("Loading images...\n")
data_test = []
data_validat = []
data_train = []
labels_test = []
labels_validat = []
labels_train = []

f=open('classification_data.dat','r')
raw_data = f.readlines()
f.close()
data_file=raw_data[1:len(raw_data)]
data_class = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
for i in range(len(data_file)):
    data_file[i] = data_file[i].split('\n')
    data_file[i] = data_file[i][0].split(',')
    
    if int(data_file[i][2]) == 1:
        aux_class=0     #good parts
    elif int(data_file[i][3]) == 1:
        aux_class=1     #no hole
    elif int(data_file[i][4]) == 1:
        aux_class=2     #have pin
    elif int(data_file[i][5]) == 1:
        aux_class=3     #broken tooth
    
    if data_file[i][1] == 'b':
        aux_color=0     #black
    elif data_file[i][1] == 'r':
        aux_color=1     #red
    elif data_file[i][1] == 's':
        aux_color=2     #silver

    data_class[aux_class][aux_color].append(data_file[i])

##------------------------------------------------------------
## Shuffle and Split data in train, validation and test set
##------------------------------------------------------------

data_test = []
data_validat = []
data_train = []
labels_test = []
labels_validat = []
labels_train = []

for i in range(len(data_class)):
    for j in range(len(data_class[i])):
        random.shuffle(data_class[i][j])
        for k in range(len(data_class[i][j])):
            image_path=('database/'+str(data_class[i][j][k][0])+image_extension)
            if k<=int(test_per*len(data_class[i][j])):
                image = cv2.imread(image_path,image_color_type)
                image = cv2.resize(image, (image_size[0], image_size[1]))
                data_test.append(image)
                labels_test.append(data_class[i][j][k])
            elif k<=int((test_per+validat_per)*len(data_class[i][j])):
                image = cv2.imread(image_path,image_color_type)
                image = cv2.resize(image, (image_size[0], image_size[1]))
                data_validat.append(image)
                labels_validat.append(data_class[i][j][k])
            else:
                image = cv2.imread(image_path,image_color_type)
                image = cv2.resize(image, (image_size[0], image_size[1]))
                data_train.append(image)
                labels_train.append(data_class[i][j][k])

##------------------------------------------------------------
## Normalise and adjust dimensions
##------------------------------------------------------------

data_test = np.array(data_test, dtype="float") / 255.0
data_validat = np.array(data_validat, dtype="float") / 255.0
data_train = np.array(data_train, dtype="float") / 255.0
labels_test = np.array(labels_test)
labels_validat = np.array(labels_validat)
labels_train = np.array(labels_train)
color_lb = LabelBinarizer()
labels_test_color = color_lb.fit_transform(labels_test[:,1])
labels_validat_color = color_lb.transform(labels_validat[:,1])
labels_train_color = color_lb.transform(labels_train[:,1])
labels_test_quality = labels_test[:,2:]
labels_validat_quality = labels_validat[:,2:]
labels_train_quality = labels_train[:,2:]

color_output_size = len(labels_train_color[0])
quality_output_size = len(labels_train_quality[0])

##----------------------------------------------------
## Create and configure color and quality CNN models
##----------------------------------------------------

train_type = input('Choose training:\n([0] No training; [1] Color only; [2] Quality only; [3] Color + Quality)\n')

if train_type == '1' or train_type == '3':

    print("------------------------------------------------------\nSetting up color ANN...\n")
    
    model_color = Sequential()
    model_color.add(Conv2D(1, filter_size, padding='same', activation='relu', input_shape=(image_size[0], image_size[1],3)))
    model_color.add(BatchNormalization())
    model_color.add(MaxPooling2D(pool_size=2))
    model_color.add(Flatten())
    model_color.add(Dense(number_dense_color, activation='relu'))
    model_color.add(BatchNormalization())
    model_color.add(Dense(color_output_size,activation='softmax'))  
    
    print("------------------------------------------------------\nTraining color ANN...\n")
    plot_model(model_color, to_file='models/color/color_ANN_model.png',show_shapes=True,show_layer_names=False)
    
    coloroptimizer = SGD(lr=lr_inicial_color, decay=lr_inicial_color/Epochs_color)
    model_color.compile(loss="categorical_crossentropy", optimizer=coloroptimizer, metrics=["accuracy"])
    colorearlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=color_earlystop_patience, verbose=0, mode='auto', restore_best_weights=True)
    callbacks_list = [colorearlystop]
    Hist_color = model_color.fit(data_train, labels_train_color, epochs=Epochs_color, validation_data=(data_validat, labels_validat_color), callbacks=callbacks_list)
    
    
    print("------------------------------------------------------\nSaving color ANN model and label binarizer...\n")
    model_color.save("models/color/color_ANN_model.h5")
    #f = open("results/color/models/color_label_binarizer.txt", "wb")
    #f.write(pickle.dumps(color_lb))
    #f.close()
    
    N = np.arange(0, (colorearlystop.stopped_epoch+1))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, Hist_color.history["acc"], label="train_acc")
    plt.plot(N, Hist_color.history["val_acc"], label="val_acc")
    plt.title("Training and validation Accuracy (color)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("models/color/color_accuracy_plot.png")
    
    print("------------------------------------------------------\nEvaluating color ANN with test data...\n")
    predict_color = model_color.predict(data_test, batch_size=Bath_size)
    print(classification_report(labels_test_color.argmax(axis=1), predict_color.argmax(axis=1), target_names=color_lb.classes_))
    
    
if train_type == '2' or train_type == '3':
    log=open('models/quality/training_log.txt','a+')
    log.write("Configurations Log:\n")
    log.write("\ntrain_per = "+str(train_per)+"\nvalidat_per = "+str(validat_per)+"\ntest_per = "+str(test_per)+
              "\nimage_color_type = "+str(image_color_type)+"\nimg_size = "+str(img_size)+"\nfilter_size = "+str(filter_size)+
              "\nnumber_dense_quality = "+str(number_dense_quality)+"\ndropout_quality = "+str(dropout_quality)+
              "\nlr_inicial_quality = "+str(lr_inicial_quality)+"\nEpochs_quality = "+str(Epochs_quality)+
              "\nquality_earlystop_patience = "+str(quality_earlystop_patience)+"\nBath_size = "+str(Bath_size)+
              "\nrandom_seed = "+str(random_seed))
    log.close()
    date_time = datetime.datetime.now()
    log=open('models/quality/training_log.txt','a+')
    log.write("\n------------------------------------------------------\nStart time: {}\nQuality learning rate = {}\n".format(date_time.strftime("%d-%m-%Y %H:%M"),lr_inicial_quality))
    log.close()
    print("\n------------------------------------------------------\nSetting up quality CNN...\n")
    print("lr =",lr_inicial_quality)
    model_quallity = Sequential()
    #[256x256x4] -> CONV2D + RELU + BatchNorm + POOL + DROP
    model_quallity.add(Conv2D(int(number_filters[0]), filter_size, padding='same', activation='relu', input_shape=(image_size[0], image_size[1],3)))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[0]), filter_size, padding='same', activation='relu', input_shape=(image_size[0], image_size[1],3)))
    model_quallity.add(BatchNormalization())
    model_quallity.add(MaxPooling2D(pool_size=2))
    #model_quallity.add(Dropout(dropout_quality))
    
    #[128x128x8] -> (CONV2D + RELU + BatchNorm)^2 + POOL + DROP
    model_quallity.add(Conv2D(int(number_filters[1]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[1]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(MaxPooling2D(pool_size=2))
    #model_quallity.add(Dropout(dropout_quality))
    
    #(64x64x16) -> (CONV2D + RELU + BatchNorm)^2 + POOL + DROP
    model_quallity.add(Conv2D(int(number_filters[2]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[2]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[2]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(MaxPooling2D(pool_size=2))
    #model_quallity.add(Dropout(dropout_quality))
    
    #(32x32x32) -> (CONV2D + RELU + BatchNorm)^3 + POOL + DROP
    model_quallity.add(Conv2D(int(number_filters[3]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[3]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[3]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(MaxPooling2D(pool_size=2))
    #model_quallity.add(Dropout(dropout_quality))
    
    model_quallity.add(Conv2D(int(number_filters[3]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[3]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(Conv2D(int(number_filters[3]), filter_size, padding='same', activation='relu'))
    model_quallity.add(BatchNormalization())
    model_quallity.add(MaxPooling2D(pool_size=2))
    #model_quallity.add(Dropout(dropout_quality))
    
    #(512x1) -> DENSE + RELU + L2 + DROP
    model_quallity.add(Flatten())
    model_quallity.add(Dense(number_dense_quality, kernel_regularizer=regularizers.l2(0.01)))
    model_quallity.add(Activation("relu"))
    model_quallity.add(BatchNormalization())
    #model_quallity.add(Dropout(dropout_quality))
    
    model_quallity.add(Dense(number_dense_quality, kernel_regularizer=regularizers.l2(0.01)))
    model_quallity.add(Activation("relu"))
    model_quallity.add(BatchNormalization())
    #model_quallity.add(Dropout(dropout_quality))
    
    model_quallity.add(Dense(number_dense_quality, kernel_regularizer=regularizers.l2(0.01)))
    model_quallity.add(Activation("relu"))
    model_quallity.add(BatchNormalization())
    #model_quallity.add(Dropout(dropout_quality))
    
    #(7x1) -> DENSE + SOFTMAX (models layer)
    model_quallity.add(Dense(quality_output_size))
    model_quallity.add(Activation("softmax"))
    
    #plot model_quallity diagram
    plot_model(model_quallity, to_file="models/quality/quality_CNN_model.png",show_shapes=True,show_layer_names=False)
    
    #seting up optimizer,agumentation and compile model_quallity
    augment = ImageDataGenerator(rotation_range=1,fill_mode="nearest") #width_shift_range=0.00, height_shift_range=0.00, shear_range=0.00, zoom_range=0.00, horizontal_flip=True, vertical_flip=True
    qualityoptimizer = SGD(lr=lr_inicial_quality, decay=lr_inicial_quality/Epochs_quality)
    model_quallity.compile(loss="categorical_crossentropy", optimizer=qualityoptimizer, metrics=["accuracy"])
    qualityearlystop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=quality_earlystop_patience, verbose=1, mode='auto',baseline=0.7, restore_best_weights=True)
    quality_callbacks_list = [qualityearlystop]
      
      
    ##----------------------------------------------------
    ## Training, save (model_quallity + label binarizer)
    ##   and plot training accuracy
    ##----------------------------------------------------
    #training model_quallity
    print("------------------------------------------------------\nTraining quality CNN...\n")
    Hist = model_quallity.fit_generator(augment.flow(data_train, labels_train_quality, batch_size=Bath_size), steps_per_epoch=len(data_train) // Bath_size, epochs=Epochs_quality, validation_data=(data_validat, labels_validat_quality), callbacks=quality_callbacks_list)
    
    # save the model_quallity and label binarizer to disk
    print("------------------------------------------------------\nSaving quality CNN model and label binarizer...\n")
    model_quallity.save("models/quality/quality_CNN_model.h5")
      
    # Plot the training accuracy
    N = np.arange(0, (qualityearlystop.stopped_epoch+1))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, Hist.history["acc"], label="train_acc")
    plt.plot(N, Hist.history["val_acc"], label="val_acc")
    plt.title("Training and validation accuracy (quality) lr="+str(lr_inicial_quality))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("models/quality/quality_loss_plot.png")
    
    ##----------------------------------------------------
    ## Predict and evaluate for test dataset
    ##----------------------------------------------------
    date_time = datetime.datetime.now()
    print("------------------------------------------------------\n","End: ",date_time.strftime("%d-%m-%Y %H:%M"),"\nEvaluating quality CNN with test data...\n")
    predict_quality = model_quallity.predict(data_test, batch_size=Bath_size)
    print(classification_report(labels_test_quality.argmax(axis=1), predict_quality.argmax(axis=1), target_names=["good part","no hole","have pin","broken tooth"]))
    log=open('models/quality/training_log.txt','a+')
    log.write(classification_report(labels_test_quality.argmax(axis=1), predict_quality.argmax(axis=1), target_names=["good part","no hole","have pin","broken tooth"]))
    log.write("\nEnd time: {}\n".format(date_time.strftime("%d-%m-%Y %H:%M")))
    log.close()
    K.clear_session()
print("------------------------------------------------------\nEnd.")
