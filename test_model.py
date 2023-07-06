from simple_multi_unet_model import multi_unet_model
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

n_classes = 4

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=256, IMG_WIDTH=512, IMG_CHANNELS=3)

model = get_model()
model.load_weights('/bess25/jskim/semantic_segmentation/U-net_colab/230706_cityscape_all_rgb.hdf5')

source_path = '/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/230706source/'
X_test = np.load(source_path + 'X_test.npy')
y_test = np.load(source_path + 'y_test.npy')
with open('names_test.pkl', 'rb') as f:
    names_test = pickle.load(f)

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3) # array 축 별로 가장 큰 값의 인덱스 뽑아줌

#Using built in keras function
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for background is: ", class1_IoU)
print("IoU for vegetation is: ", class2_IoU)
print("IoU for sidewalk is: ", class3_IoU)
print("IoU for road is: ", class4_IoU)

# plt.imshow(train_images[0, :,:,0], cmap='gray')
# plt.imshow(train_masks[0], cmap='gray')

#Predict on a few images
#model = get_model()
#model.load_eights('???.hdf5') 
prediction_path = '/bess25/jskim/semantic_segmentation/U-net_colab/result/230706predictions/array'
figure_path = '/bess25/jskim/semantic_segmentation/U-net_colab/result/230706predictions/figure'
for i in range(len(X_test)):
    test_img = X_test[i]
    print(test_img)
    ground_truth=y_test[i]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]


    # Remove extension from filename
    name = names_test[i].split(".")[0]

    # np.save
    np.save(os.path.join(prediction_path, f'{name}.npy'), predicted_img)

    plt.figure(figsize=(12, 4))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    # plt.show()
    plt.savefig(os.path.join(figure_path, f'prediction_{name}.png'))
