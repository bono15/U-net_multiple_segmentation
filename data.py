from simple_multi_unet_model import multi_unet_model #Uses softmax 

from PIL import Image
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

class data_loader():
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path

    def dataload(self):
        #Resizing images, if needed
        SIZE_X = 512 
        SIZE_Y = 256
        n_classes=4 #Number of classes for segmentation

        #Capture training image info as a list
        train_images = []
        directory_path=self.img_path
        for img in glob.glob(os.path.join(directory_path, "*.png")):
            img = cv2.imread(img, 0)       
            #img = cv2.resize(img, (SIZE_Y, SIZE_X))
            train_images.append(img)
            
        #Convert list to array for machine learning processing        
        train_images = np.array(train_images)

        #Capture mask/label info as a list
        train_masks = [] 
        directory_path= self.mask_path
        for mask in glob.glob(os.path.join(directory_path, "*.png")):
            mask = cv2.imread(mask, 0)       
            #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            train_masks.append(mask)
                
        #Convert list to array for machine learning processing          
        train_masks = np.array(train_masks)

        #Encode labels... but multi dim array so need to flatten, encode and reshape
        
        labelencoder = LabelEncoder()
        n, h, w = train_masks.shape
        train_masks_reshaped = train_masks.reshape(-1,1)
        train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
        train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

        np.unique(train_masks_encoded_original_shape)

        #################################################
        train_images = np.expand_dims(train_images, axis=3) #딥러닝에 맞는 형태로 변형(num_samples, height, width, channels=1)
        train_images = normalize(train_images, axis=1) #height, width 정보 남기고 눌러줌

        train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)



        #Create a subset of data for quick testing
        #Picking 10% for testing and remaining for training
        from sklearn.model_selection import train_test_split #>scikit learn은 데이터 분석 툴, train_test_split라는 기능이 있음. X는 데이터, y는 레이블. 
        X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

        #Further split training data to a smaller subset for quick testing of models
        X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

        print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

        
        train_masks_cat = to_categorical(y_train, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

        test_masks_cat = to_categorical(y_test, num_classes=n_classes)
        y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
        print(y_train_cat.shape)
        print(y_test_cat.shape)


        ###############################################################
        

        # source_path = '/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/230623source/'
        # np.save(source_path + 'X_train.npy', X_train)
        # np.save(source_path + 'X_test.npy', X_test)
        # np.save(source_path + 'y_train.npy', y_train)
        # np.save(source_path + 'y_test.npy', y_test)
        # np.save(source_path + 'y_train_cat.npy', y_train_cat)
        # np.save(source_path + 'y_test_cat.npy', y_test_cat)


        self.class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                        classes=np.unique(train_masks_reshaped_encoded), 
                                                        y=train_masks_reshaped_encoded)
        # print("Class weights are...:", class_weights)

        IMG_HEIGHT = X_train.shape[1]
        print(IMG_HEIGHT)
        IMG_WIDTH  = X_train.shape[2]
        print(IMG_WIDTH)
        IMG_CHANNELS = X_train.shape[3]
        print(IMG_CHANNELS)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS

        # def get_model():
        #     return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

        # model = get_model()
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary()

        # print(model.summary())

        # history = model.fit(X_train, y_train_cat, 
        #                     batch_size = 16, 
        #                     verbose=1, 
        #                     epochs=50, 
        #                     validation_data=(X_test, y_test_cat), 
        #                     # class_weight=class_weights, #이미지따라
        #                     shuffle=False)
                        
        # model.save('/bess25/jskim/semantic_segmentation/U-net_colab/cityscape_all.hdf5')


        # #model = get_model()
        # model.load_weights('/bess25/jskim/semantic_segmentation/U-net_colab/cityscape_all.hdf5')  

        # #IOU
        # y_pred=model.predict(X_test)
        # y_pred_argmax=np.argmax(y_pred, axis=3)

        #Predict on a few images
        #model = get_model()
        #model.load_weights('???.hdf5')  
        import random
        test_img_number = random.randint(0, len(X_test))
        test_img = X_test[test_img_number]
        ground_truth=y_test[test_img_number]
        test_img_norm=test_img[:,:,0][:,:,None]
        test_img_input=np.expand_dims(test_img_norm, 0)
        # prediction = (model.predict(test_img_input))
        # predicted_img=np.argmax(prediction, axis=3)[0,:,:]
        return X_train,X_test,y_train,y_test,y_train_cat,y_test_cat,test_img_input,ground_truth
