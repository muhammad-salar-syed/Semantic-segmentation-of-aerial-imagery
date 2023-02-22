
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm
from smooth_blending import predict_img_with_smooth_windowing
from Metrics import Jacard_coefficient 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

import random
import glob

img_path=glob.glob('./Semantic segmentation dataset/Tile1/images/*')
mask_path=glob.glob('./Semantic segmentation dataset/Tile1/masks/*')

num=random.randint(0,len(img_path)-1)
img = cv2.imread(img_path[num])
mask = cv2.cvtColor(cv2.imread(mask_path[num]),cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
print('img shape:',img.shape)
plt.subplot(122)
plt.imshow(mask)
print('mask shape:',mask.shape)
plt.show()

from keras.models import load_model
model = load_model("./satalite.hdf5", compile=False)

patch_size = 256
n_classes = 6

# Predict on patch 
SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by 256
SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by 256
large_img = Image.fromarray(img)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
large_img = np.array(large_img)  
print(large_img.shape)

patch_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #no overlap
patch_img = patch_img[:,:,0,:,:,:]
print(patch_img.shape)


# for plotting
from preprocessing import rgb_label
label = Image.fromarray(mask)
label = label.crop((0 ,0, SIZE_X, SIZE_Y))
label = np.array(label)  
label = rgb_label(label)  
print(label.shape) 
plt.imshow(label)
plt.imshow(mask)
 


prediction = []
for i in range(patch_img.shape[0]):
    for j in range(patch_img.shape[1]):
        
        single_patch_img = patch_img[i,j,:,:,:]
        
        # Using minmaxscaler for normalization
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]
                                 
        prediction.append(pred)
        
patched_prediction = np.array(prediction)
patched_prediction = np.reshape(patched_prediction, [patch_img.shape[0], patch_img.shape[1], 
                                            patch_img.shape[2], patch_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))


plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(large_img)
plt.axis('off')
print('img shape:',large_img.shape)
plt.subplot(132)
plt.imshow(label)
print('label shape:',label.shape)
plt.axis('off')
plt.subplot(133)
plt.imshow(unpatched_prediction)
print('unpatched_prediction shape:',unpatched_prediction.shape)
plt.axis('off')
plt.show()

#############################################################

#Predict using smooth blending

input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict((img_batch_subdiv))
    )
)

final_prediction = np.argmax(predictions_smooth, axis=2)

#Convert labeled images back to original RGB colored masks. 

def label_to_rgb(predicted_image):
    
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

#Plot and save results
prediction_with_smooth_blending=label_to_rgb(final_prediction)
prediction_without_smooth_blending=label_to_rgb(unpatched_prediction)


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Testing Label')
plt.imshow(mask)
plt.subplot(223)
plt.title('Prediction without smooth blending')
plt.imshow(prediction_without_smooth_blending)
plt.subplot(224)
plt.title('Prediction with smooth blending')
plt.imshow(prediction_with_smooth_blending)
plt.savefig('./prediction_with_smooth_blending.png')