
import glob
import cv2
import numpy as np
path1=glob.glob('./saved_data/images/*.*')
path2=glob.glob('./saved_data/masks/*.*')

images=[]
for i in range(len(path1)):
    img=cv2.imread(path1[i])/255.
    images.append(img)
    
images_array=np.array(images)


masks=[]
for i in range(len(path2)):
    m=cv2.imread(path2[i],0)
    masks.append(m)
    
masks_array=np.array(masks)  
masks_array = np.expand_dims(masks_array, axis=3)
print("Unique labels: ", np.unique(masks_array))

import random
import numpy as np
import matplotlib.pyplot as plt
num = random.randint(0, len(images_array))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(images_array[num])
plt.subplot(122)
plt.imshow(masks_array[num][:,:,0])
plt.show()

classes = len(np.unique(masks_array))
from keras.utils import to_categorical
import segmentation_models as sm
labels_cat = to_categorical(masks_array, num_classes=classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images_array, labels_cat, test_size = 0.20, random_state = 42)

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from SingleChannel_multiclass_Unet import Unet  
from Metrics import Jacard_coefficient

metrics=['accuracy', Jacard_coefficient]

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
early_stop = EarlyStopping(monitor='val_loss', patience=12, verbose=1)
log_csv = CSVLogger('./satalite_logs.csv', separator=',', append=False)
callbacks_list = [early_stop, log_csv]

model =Unet(n_classes=classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks=callbacks_list)
model.save('./satalite.hdf5')