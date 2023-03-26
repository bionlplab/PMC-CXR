import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from model_binary import Den,Res
import numpy as np
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import copy
import cv2
from skimage.transform import resize
import copy
from keras.utils import np_utils
import matplotlib.pyplot as plt
image_path = '/prj0129/mil4012/glaucoma/PMCFigureX'
label_path = '/prj0129/mil4012/glaucoma/PMCFigureX/normal_cxr_ct_training/normal_cxr_ct.csv'

def weighted_binary_crossentropy(y_true, y_pred) :
#     weight = 1 - K.sum(y_true) /(K.sum(y_true) + K.sum(1 - y_true))
    weight = 0.9
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight +  (1 - y_true) * K.log(1 - y_pred) * (1-weight))
    return K.mean(logloss, axis=-1)


def train(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    datagen.fit(x_train)
#     print('data shape of x_train is', np.shape(x_train), np.shape(y_train))
    print('data shape of x_train is', type(x_train), type(y_train))
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    print('the program start to fit')
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size= 64), validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // 64, epochs=epochs
                        , shuffle=True, callbacks=[model_checkpoint])
    
    
    accy = history.history['accuracy']
    lossy = history.history['loss']
    np_accy = np.array(accy).reshape((1,len(accy)))
    np_lossy = np.array(lossy).reshape((1,len(lossy)))
    np_out = np.concatenate([np_accy,np_lossy],axis=0)
    np.savetxt('/prj0129/mil4012/glaucoma/PMCFigureX/train.txt',np_out)
    
    
    accy_val = history.history['val_accuracy']
    lossy_val = history.history['val_loss']
    np_accy_val = np.array(accy_val).reshape((1,len(accy_val)))
    np_lossy_val = np.array(lossy_val).reshape((1,len(lossy_val)))
    np_out_val = np.concatenate([np_accy_val,np_lossy_val],axis=0)
    np.savetxt('/prj0129/mil4012/glaucoma/PMCFigureX/val.txt',np_out_val)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig('/prj0129/mil4012/glaucoma/PMCFigureX/accuracy1.jpg')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig('/prj0129/mil4012/glaucoma/PMCFigureX/loss1.jpg')
    
#     model.fit(x_train,y_train, validation_data=(x_val, y_val), batch_size= 64, epochs=epochs
#               ,shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')
    
    
# def test(x_test, y_test, model, weights):
# #def test(x_test, y_test, model, weights):
#     model.load_weights(weights)
#     p_test = model.predict(x_test)
#     p_classes = copy.deepcopy(p_test)
# #     y_pred = np.argmax(p_classes, axis=0)
# #     y_test = np.argmax(y_test, axis=0)
#     print('the shape of p_test',np.shape(p_test))
#     print('the shape of y_test',np.shape(y_test))
#     y_pred = np.argmax(p_classes, axis=1)
#     y_test = np.argmax(y_test, axis=1)
#     print('the shape of p_classes',np.shape(y_pred))
#     print('the shape of y_test',np.shape(y_test))
#     target_names = ['cxr', 'others']
#     print(classification_report(y_test, y_pred, target_names=target_names,digits=2))
    

    
#     return


def test(x_test, y_test, model, weights):
#def test(x_test, y_test, model, weights):
    model.load_weights(weights)
    p_test = model.predict(x_test)
    # np.savetxt(weights[:-3]+'.txt', np.reshape(p_test,(len(p_test),)))
#    p_test = get_test1(x_test, y_test, model, weights)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]
#    print(p_test)
#    print(p_classes)
    print('the shape of test is', p_test.shape)
    accuracy = accuracy_score(y_test, p_classes)
    print('classification accuracy: ', accuracy)
    precision = precision_score(y_test, p_classes)
    print('precision: ', precision)
    recall = recall_score(y_test, p_classes)
    print('recall: ', recall)
    f1 = f1_score(y_test, p_classes)
    print('F1 score: ', f1)
    auc = roc_auc_score(y_test, p_test)
    print('AUC: ', auc)
    matrix = confusion_matrix(y_test, p_classes)
    print(matrix)

    
    return


if __name__ == '__main__':
#     model = Den(den_en='den121',img_size=(224, 224, 3), dropout=False)
    model = Res(res_en='res50',img_size=(224, 224, 3), dropout=False)
    learning_rate = 5*1e-5
    epochs = 3
    weights_path = 'cxr_resnet50new.h5'
    
    train_img = []
    train_label = []
    val_img = []
    val_label =[]
    test_img = []
    test_label = []
    # get the data from normal_cxr_ct_training, 500, 500, and 500 for traning, validation,and testing, respectively
    tmp = np.loadtxt(label_path, dtype=np.str, delimiter=",")
    image_index = tmp[:,0]
    image_index = image_index[1:len(image_index)]
    labels = tmp[:,1]
    labels = labels[1:len(labels)]
    fold = tmp[:,2]
    fold = fold[1:len(fold)]
    tot = np.zeros((3,2))
    
    for i in range(len(fold)):
        if fold[i] == 'train':
            im = cv2.imread(os.path.join(image_path,'normal_cxr_ct_training/images',image_index[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            if labels[i] == 'cxr':
                train_label.append(1)
                tot[0,0] += 1
            else:
                train_label.append(0)
                tot[0,1] += 1
        
        if fold[i] == 'val':
            im = cv2.imread(os.path.join(image_path,'normal_cxr_ct_training/images',image_index[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            if labels[i] == 'cxr':
                val_label.append(1)
                tot[1,0] += 1
            else:
                val_label.append(0)
                tot[1,1] += 1
                
                
        if fold[i] == 'test':
            im = cv2.imread(os.path.join(image_path,'normal_cxr_ct_training/images',image_index[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            if labels[i] == 'cxr':
                test_label.append(1)
                tot[2,0] += 1
            else:
                test_label.append(0)
                tot[2,1] += 1
    
    print('the tot is', tot)
    ##loading data from litcovid_data folder
    ##loading ct data from ct folder
    files = os.listdir(os.path.join(image_path,'litcovid_data/ct'))
    N = len(files)
    print('the length of N is', N)
    for i in range(N):
        if i < int(int(N*0.7)):
            im = cv2.imread(os.path.join(image_path,'litcovid_data/ct',files[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            train_label.append(0)
        elif i >= int(int(N*0.7)) and i < int(int(N*0.8)):
            im = cv2.imread(os.path.join(image_path,'litcovid_data/ct',files[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            val_label.append(0)
        else:
            im = cv2.imread(os.path.join(image_path,'litcovid_data/ct',files[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            test_label.append(0)
        
        
    ##loading cxr data from cxr folder
    files = os.listdir(os.path.join(image_path,'litcovid_data/cxr'))
    N = len(files)
    print('the length of N is', N)
    for i in range(N):
        if i < int(int(N*0.7)):
            im = cv2.imread(os.path.join(image_path,'litcovid_data/cxr',files[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            train_label.append(1)
        elif i >= int(int(N*0.7)) and i < int(int(N*0.8)):
            im = cv2.imread(os.path.join(image_path,'litcovid_data/cxr',files[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            val_label.append(1)
        else:
            im = cv2.imread(os.path.join(image_path,'litcovid_data/cxr',files[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            test_label.append(1)
        
     
    ##loading ct data from ct folder
    files = os.listdir(os.path.join(image_path,'litcovid_data/lesion_ct'))
    N = len(files)
    print('the length of N is', N)
    for i in range(N):
        if i < int(int(N*0.7)):
            im = cv2.imread(os.path.join(image_path,'litcovid_data/lesion_ct',files[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            train_label.append(0)
        elif i >= int(int(N*0.7)) and i < int(int(N*0.8)):
            im = cv2.imread(os.path.join(image_path,'litcovid_data/lesion_ct',files[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            val_label.append(0)
        else:
            im = cv2.imread(os.path.join(image_path,'litcovid_data/lesion_ct',files[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            test_label.append(0)
        
        
    
    
    ##laoding normal case from doc folder
    files = os.listdir(os.path.join(image_path,'doc'))
    N = len(files)
    print('the length of N is', N)
    for i in range(N):
        if i < int(int(N*0.7)):
            im = cv2.imread(os.path.join(image_path,'doc',files[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            train_label.append(0)
        elif i >= int(int(N*0.7)) and i < int(int(N*0.8)):
            im = cv2.imread(os.path.join(image_path,'doc',files[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            val_label.append(0)
        else:
            im = cv2.imread(os.path.join(image_path,'doc',files[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            test_label.append(0)
    
    
    
    ##laoding normal case from Clef2016 folder
    files = os.listdir(os.path.join(image_path,'Clef2016'))
    N = len(files)
    print('the length of N is', N)
    for i in range(N):
        if i < int(int(N*0.7)):
            im = cv2.imread(os.path.join(image_path,'Clef2016',files[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            train_label.append(0)
        elif i >= int(int(N*0.7)) and i < int(int(N*0.8)):
            im = cv2.imread(os.path.join(image_path,'Clef2016',files[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            val_label.append(0)
        else:
            im = cv2.imread(os.path.join(image_path,'Clef2016',files[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            test_label.append(0)
    
    
    ##laoding cxr case from NIH-chest-xfolder
    files = os.listdir(os.path.join(image_path,'NIH-chest-x'))
    N = len(files)
    print('the length of N is', N)
    for i in range(N):
        if i < int(int(N*0.7)):
            im = cv2.imread(os.path.join(image_path,'NIH-chest-x',files[i]))
            im = cv2.resize(im,(224,224))
            train_img.append(im)
            train_label.append(1)
        elif i >= int(int(N*0.7)) and i < int(int(N*0.8)):
            im = cv2.imread(os.path.join(image_path,'NIH-chest-x',files[i]))
            im = cv2.resize(im,(224,224))
            val_img.append(im)
            val_label.append(1)
        else:
            im = cv2.imread(os.path.join(image_path,'NIH-chest-x',files[i]))
            im = cv2.resize(im,(224,224))
            test_img.append(im)
            test_label.append(1)
    
    train_img = np.array(train_img)
    val_img = np.array(val_img)
    test_img = np.array(test_img)
    
    
    # train_label = train_label.astype(np.float)
    # val_label = val_label.astype(np.float)
    # test_label = test_label.astype(np.float) 
    
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label) 
    
    print('the length of training', np.shape(train_label))
    print('the length of validation', np.shape(val_label))
    print('the length of testing', np.shape(test_label))
    
    
    print('the shape of training', np.shape(train_img))
    print('the shape of validation', np.shape(val_img))
    print('the shape of testing', np.shape(test_img))




    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
    
    
    
    train(train_img, train_label, val_img, val_label, model, epochs, weights_path)
    test(test_img, test_label, model, weights_path)