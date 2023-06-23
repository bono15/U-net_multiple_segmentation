from data import train_model
import numpy as np
import os
from matplotlib import pyplot as plt

source_path = '/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/230623source/'
X_train = np.load(source_path + 'X_train.npy')
X_test = np.load(source_path + 'X_test.npy')
y_train = np.load(source_path + 'y_train.npy')
y_test = np.load(source_path + 'y_test.npy')
y_train_cat = np.load(source_path + 'y_train_cat.npy')
y_test_cat = np.load(source_path + 'y_test_cat.npy')

# def train_model(X_train, y_train_cat, X_test, y_test_cat, model):
#     history = model.fit(X_train, y_train_cat, 
#                     batch_size = 16, 
#                     verbose=1, 
#                     epochs=50, 
#                     validation_data=(X_test, y_test_cat), 
#                     # class_weight=class_weights, #이미지따라
#                     shuffle=False)
#     return model, history
n_classes=4

model, class_weights = train_model(X_train, y_train, n_classes)

history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test_cat), 
                    # class_weight=class_weights, #이미지따라
                    shuffle=False)
                   
model.save('/bess25/jskim/semantic_segmentation/U-net_colab/230623_cityscape_all.hdf5')

#Evaluate the model
	# evaluate model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/bess25/jskim/semantic_segmentation/U-net_colab/result/230623_cityscape_all_loss.png')
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/bess25/jskim/semantic_segmentation/U-net_colab/result/230623_cityscape_all_accuracy.png')
plt.show()

#model = get_model()
model.load_weights('/bess25/jskim/semantic_segmentation/U-net_colab/230623_cityscape_all.hdf5')

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 4
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

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

# plt.imshow(train_images[0, :,:,0], cmap='gray')
# plt.imshow(train_masks[0], cmap='gray')

#Predict on a few images
#model = get_model()
#model.load_weights('???.hdf5')  
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.savefig('/bess25/jskim/semantic_segmentation/U-net_colab/result/230623_cityscape_all_prediction.png')
plt.show()
