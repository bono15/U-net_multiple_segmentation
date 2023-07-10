from data import data_loader
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
from simple_multi_unet_model import multi_unet_model

n_classes=4

if __name__ == '__main__':
    img_path  ="/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/image_RGB"
    mask_path = "/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/mask_4classes"
    data_loader = data_loader(img_path,mask_path)
    X_train,X_test,y_train,y_test,y_train_cat,y_test_cat,names_train,names_test = data_loader.dataload()


    # Save X_test, y_test
    source_path = '/bess25/jskim/semantic_segmentation/U-net_colab/DLsource/230710source/'
    np.save(os.path.join(source_path, 'X_test.npy'), X_test)
    np.save(os.path.join(source_path + 'y_test.npy'), y_test)

    # Save names_test
    with open(os.path.join(source_path, 'names_test.pkl'), 'wb') as f:
        pickle.dump(names_test, f)

    class_weights = data_loader.class_weights
    IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS = data_loader.IMG_HEIGHT,data_loader.IMG_WIDTH,data_loader.IMG_CHANNELS

    def get_model():
        return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train_cat, 
                        batch_size = 16, 
                        verbose=1, 
                        epochs=50, 
                        validation_data=(X_test, y_test_cat), 
                        # class_weight=class_weights, #이미지따라
                        shuffle=False)
                    
    model.save('/bess25/jskim/semantic_segmentation/U-net_colab/230710_cityscape_all_rgb.hdf5')

    #Evaluate the model
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
    plt.savefig('/bess25/jskim/semantic_segmentation/U-net_colab/result/230710_cityscape_all_loss.png')
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/bess25/jskim/semantic_segmentation/U-net_colab/result/230710_cityscape_all_accuracy.png')
    plt.show()

