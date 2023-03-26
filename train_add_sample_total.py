import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from model import Den,Res
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
import copy
import cv2
from skimage.transform import resize
import copy
import re
import pandas as pd
image_path_train = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/train_cxr.csv'
image_path_val = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/val_cxr.csv'
image_path_test = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/test_cxr.csv'

image_path = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/images/images'
# label_path = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/label.csv'
label_path = '/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/nih-label-v1.csv'

def weighted_binary_crossentropy(y_true, y_pred) :
#     weight = 1 - K.sum(y_true) /(K.sum(y_true) + K.sum(1 - y_true))
    weight = 0.9
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight +  (1 - y_true) * K.log(1 - y_pred) * (1-weight))
    return K.mean(logloss, axis=-1)


def data_generator(train_path, train_labels, datagens, batch_size=64):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(train_labels)))
        num_batches = len(train_labels) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
           # batch = [train_generator[i] for i in batch_indices]
            X = np.zeros((batch_size, 224, 224, 3))
            Y = np.zeros((batch_size, ))
            for i in range(batch_size):
                if datagens is None:
                    im = cv2.imread(train_path[batch_indices[i]])            
                    # X[i] = cv2.resize(im,(224,224))
                else:
                    im = cv2.imread(train_path[batch_indices[i]]) 
                    X[i] = datagens.random_transform(cv2.resize(im,(224,224)))
                    # X[i] = datagens.random_transform(im)
                Y[i] = train_labels[[batch_indices[i]]]
            yield X, Y

def train1(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    
    datagen_args = dict(rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True)
    datagens = ImageDataGenerator(**datagen_args)
    BATCH_SIZE = 96
    train_gen = data_generator(x_train, y_train, datagens, BATCH_SIZE)

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    model.fit_generator(train_gen, validation_data=(x_val, y_val), steps_per_epoch=len(y_train) // BATCH_SIZE, epochs=epochs
              ,shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')

def train(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    datagen.fit(x_train)
#    print('data tpye of x_train is', type(x_train), type(y_train))
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    print('the program start to fit')
    model.fit_generator(datagen.flow(x_train, y_train, batch_size= 64), validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // 64, epochs=epochs
                        , shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')
    
    
def test(x_test, y_test, model, weights):
#def test(x_test, y_test, model, weights):
    model.load_weights(weights)
    p_test = model.predict(x_test)
#    np.savetxt(weights[i][:-3]+'.txt', np.reshape(p_test,(len(p_test),)))
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
    spec = matrix[0,0]/(matrix[0,0] + matrix[0,1])
    print('specificity: ', spec)

    
    return



if __name__ == '__main__':
    model = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
#     model = Res(res_en='res50',img_size=(224, 224, 3), dropout=False)
    learning_rate = 5*1e-5
    epochs = 15
    weights_path = 'Pneumonia_cxr_den201_add_nih_postive1.h5'
    
    #get data and label for training, validate, and testing dataset.
    
    #training dataset
    tmp = np.loadtxt(image_path_train, dtype=np.str, delimiter=",")
    train_path = tmp[:,0]
    train_path = train_path[1:len(train_path)]
    # train_path = train_path[1:2000]
    #7-> lung lesion, 12->pneumonia, 13->pneumothorax
    labels = tmp[:,12]
    print('the disease is',labels[0])
    labels = labels[1:len(labels)]  
    # labels = labels[1:2000]
    train_label = []
    for i in range(len(labels)):
        if labels[i] == '1':
            train_label = np.append(train_label,1)
        else:
            train_label = np.append(train_label,0)
    
    #val dataset
    tmp = np.loadtxt(image_path_val, dtype=np.str, delimiter=",")
    val_path = tmp[:,0]
    val_path = val_path[1:len(val_path)]
    #7-> lung lesion, 12->pneumonia, 13->pneumothorax
    labels = tmp[:,12]
    print('the disease is',labels[0])
    labels = labels[1:len(labels)]  
    
    val_img = np.zeros((len(labels),224,224,3))
    val_label = []
    for i in range(len(labels)):
        val_img[i,:,:,:] = cv2.imread(val_path[i])
        if labels[i] == '1.0':
            val_label = np.append(val_label,1)
        else:
            val_label = np.append(val_label,0)
    
    
        
    #test dataset
    tmp = np.loadtxt(image_path_test, dtype=np.str, delimiter=",")
    test_path = tmp[:,0]
    test_path = test_path[1:len(test_path)]
    #7-> lung lesion, 12->pneumonia, 13->pneumothorax
    labels = tmp[:,12]
    print('the disease is',labels[0])
    labels = labels[1:len(labels)]  
    
    test_img = np.zeros((len(labels),224,224,3))
    test_label = []
    for i in range(len(labels)):
        test_img[i,:,:,:] = cv2.imread(test_path[i])
        if labels[i] == '1.0':
            test_label = np.append(test_label,1)
        else:
            test_label = np.append(test_label,0)
                        
    train_label = train_label.astype(np.float)
    val_label = val_label.astype(np.float) 
    test_label = test_label.astype(np.float) 
    print('the number of lung cancer in train set', len(np.argwhere(train_label==1)))
    print('the number of lung cancer in validate set', len(np.argwhere(val_label==1)))
    print('the number of lung cancer in test set', len(np.argwhere(test_label==1)))
 
    
###generate additional data from NIH

  #get label
    tmp_nih = np.loadtxt(label_path, dtype=np.str, delimiter=",")
    image_index = tmp_nih[:,0]
    image_index = image_index[1:len(image_index)]
    #13->pneumonia, 14->pneumothorax
    nih_labels = tmp_nih[:,13]
    print('the disease is',nih_labels[0])
    nih_labels = nih_labels[1:len(nih_labels)]
    
    gt = nih_labels
    
    train_val_list = np.loadtxt('/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/train_val_list.txt', dtype=np.str)
    
    # train_val_img = np.zeros((len(train_val_list),224,224,3))
    train_val_path = []
    train_val_label = []
    
    
    for i in range(len(train_val_list)):
    # for i in range(2000):
        # im = cv2.imread(os.path.join(image_path,train_val_list[i]))
        # train_val_img[i,:,:,:] = cv2.resize(im,(224,224))
        train_val_path = np.append(train_val_path,os.path.join(image_path,train_val_list[i]))
        ind = np.argwhere(image_index==train_val_list[i])[0][0]
        train_val_label = np.append(train_val_label,gt[ind])
    
    train_val_label = train_val_label.astype(np.float)
    print('the number of Hernia in training and validation set', len(np.argwhere(train_val_label==1)))
    print('the type of disease in train and validation set is',np.unique(train_val_label))
    # generate random index to split training dataset and validation dataset
    
    index_t = np.loadtxt('ind_train.txt')
    # index_t = np.arange(0,1600)
    index_t = index_t.astype(int)
    index_v = np.loadtxt('ind_val.txt')
    # index_v = np.arange(1600,2000)
    index_v = index_v.astype(int)
    
#     index = np.random.choice(len(train_val_label),len(train_val_label),replace=False)
#     index_t = index[0:int(len(index)*0.8)]
#     index_v = index[int(len(index)*0.8):]
    
#     np.savetxt('ind_train.txt', np.reshape(index_t,(len(index_t),)))
#     np.savetxt('ind_val.txt', np.reshape(index_v,(len(index_v),)))
    
    # train_img = train_val_img[index_t,:,:,:]
    train_path_nih = train_val_path[index_t]
    train_label_nih = train_val_label[index_t]
    
    inde = np.argwhere(train_label_nih==1)
    inde = np.reshape(inde,(len(inde),))
    
    train_path_nih = train_path_nih[inde]
    train_path_nih = train_path_nih.tolist()
    train_label_nih = np.zeros((len(inde),)) + 1
    
    # val_img = train_val_img[index_v,:,:,:]
    val_path_nih = train_val_path[index_v]
    val_label_nih = train_val_label[index_v]
    
    
    inde = np.argwhere(val_label_nih==1)
    inde = np.reshape(inde,(len(inde),))
    
    val_path_nih = val_path_nih[inde]
    val_path_nih = val_path_nih.tolist()
    val_label_nih = np.zeros((len(inde),)) + 1
    val_img_nih = np.zeros((len(inde),224,224,3))
    
    # val_img_nih = np.zeros((len(val_path_nih),224,224,3))
    for i in range(len(val_path_nih)):
        im = cv2.imread(val_path_nih[i])
        val_img_nih[i,:,:,:] = cv2.resize(im,(224,224))
    
    train_path = np.concatenate((train_path,train_path_nih),axis=0)
    train_label = np.concatenate((train_label,train_label_nih),axis=0)
    
    val_img = np.concatenate((val_img,val_img_nih),axis=0)
    val_label = np.concatenate((val_label,val_label_nih),axis=0)
    
    print('the number of positive case of training data from nih', len(train_label_nih))
    print('the number of positive case of validation data from nih', len(val_label_nih))
    print('the length of training seting', len(train_label))
        
    
### generate additional data from pubmed   
    img_hernia_path = '/prj0129/mil4012/glaucoma/PMCFigureX/bioc_Pneumonia'
#     hernia_path = '/prj0129/mil4012/glaucoma/PMCFigureX/Hernia/Hernia.figures_pred.csv'
#     hernia_path1 = '/prj0129/mil4012/glaucoma/PMCFigureX/Hernia/Hernia.subfigures_pred.csv'
    hernia_path = '/prj0129/mil4012/glaucoma/PMCFigureX/Pneumonia/Pneumonia.figures_pred_final.csv'
    hernia_path1 = '/prj0129/mil4012/glaucoma/PMCFigureX/Pneumonia/Pneumonia.subfigures_pred_final.csv'
    hernia_result_path = '/prj0129/mil4012/glaucoma/PMCFigureX/Pneumonia/Pneumonia_results.csv'
    tmp = np.loadtxt(hernia_path, dtype=np.str, delimiter=",")
    predict = tmp[:,1]
    predict = predict[1:len(predict)]
#     predict_v = np.zeros(len(predict))

#     figure_path = tmp[:,1]
    figure_path = tmp[:,0]
    figure_path = figure_path[1:len(figure_path)]
    
    tmp1 = np.loadtxt(hernia_path1, dtype=np.str, delimiter=",")
#     predict1 = tmp1[:,10]
    predict1 = tmp1[:,1]
    predict1 = predict1[1:len(predict1)]
#     figure_path1 = tmp1[:,1]
    figure_path1 = tmp1[:,0]
    figure_path1 = figure_path1[1:len(figure_path1)]
    
    tmp_result = np.loadtxt(hernia_result_path, dtype=np.str, delimiter=",")
    text_result = tmp_result[:,1]
    print('the disease is',text_result[0])
    text_result = text_result[1:len(text_result)]

            
    
    
    pid = tmp_result[:,0]
    pid = pid[1:len(pid)]
    
    img = np.zeros((1000,224,224,3))
    img_path = []
    disease_gt = []
    le = 0
    tot = 0
    tot_s = 0
    
    figure_path2 = copy.deepcopy(figure_path1)
    figure_path2 = ' '.join(figure_path2)
    for i in range(len(predict)):
        if predict[i] == 'cxr':
            if figure_path[i][:-4] in figure_path2:
                k = 0
            else:
                ind = np.argwhere(figure_path[i]==pid)
                if text_result[ind[0][0]] == 'p':
                    im = cv2.imread(os.path.join(img_hernia_path,figure_path[i]))
                    img[le,:,:,:] = cv2.resize(im,(224,224))
                    img_path = np.append(img_path,os.path.join(img_hernia_path,figure_path[i]))
                    le += 1
                    disease_gt = np.append(disease_gt,1)
                    tot += 1

    for i in range(len(predict1)):
        if predict1[i] == 'cxr':
            im = cv2.imread(os.path.join(img_hernia_path,figure_path1[i]))
            img_size = np.shape(im)
            ind1 = [substr.start() for substr in re.finditer('_',figure_path1[i])]
            fig_path = figure_path1[i][:ind1[-2]] + '.jpg'
            ind = np.argwhere(fig_path==pid)
            if img_size[0] * img_size[1] > (np.square((img_size[0] + img_size[1])))*2/9 and text_result[ind[0][0]] == 'p':
                img[le,:,:,:] = cv2.resize(im,(224,224))
                img_path = np.append(img_path,os.path.join(img_hernia_path,figure_path1[i]))
                le += 1
                disease_gt = np.append(disease_gt,1)
                
                tot_s += 1

    img = img[0:le,:,:,:]
    print('the shape of img',np.shape(img))
    print('the type of disease is',np.unique(disease_gt))
    
    
    
    
    # add hernia data
    train_path = np.concatenate((train_path,img_path[0:int(len(disease_gt)*0.8)]),axis=0)
    # train_img = np.concatenate((train_img,img[0:int(len(disease_gt)*0.8),:,:,:]),axis=0)
    train_label = np.concatenate((train_label,disease_gt[0:int(len(disease_gt)*0.8)]),axis=0)
    
    val_img = np.concatenate((val_img,img[int(len(disease_gt)*0.8):,:,:,:]),axis=0)
    val_label = np.concatenate((val_label,disease_gt[int(len(disease_gt)*0.8):]),axis=0)

    
    model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_binary_crossentropy)
    
    
#     train(img, hernia_gt, img, hernia_gt, model, epochs, weights_path)
#     test(img, hernia_gt, model, weights_path)
    
    
    train1(train_path, train_label, val_img, val_label, model, epochs, weights_path)
    test(test_img, test_label, model, weights_path)