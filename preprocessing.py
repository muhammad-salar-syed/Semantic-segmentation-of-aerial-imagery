
'''
The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. The classes are:
Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B

Using patchify library to create smaller frames from larger tile image
Tile 1: 797 x 644 --> 768 x 512 --> 6
Tile 2: 509 x 544 --> 512 x 256 --> 2
Tile 3: 682 x 658 --> 512 x 512  --> 4
Tile 4: 1099 x 846 --> 1024 x 768 --> 12
Tile 5: 1126 x 1058 --> 1024 x 1024 --> 16
Tile 6: 859 x 838 --> 768 x 768 --> 9
Tile 7: 1817 x 2061 --> 1792 x 2048 --> 56
Tile 8: 2149 x 1479 --> 1280 x 2048 --> 40
Total 9 images in each folder * (145 patches) = 1305
Total 1305 patches of size 256x256
'''
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

root_dir = './Semantic segmentation dataset/'
size = 256
 
# Images
processed_data = []  
for root, subdirs, files in os.walk(root_dir): 
    dirname = root.split(os.path.sep)[-1]
    #print(dirname)
    if dirname == 'images':   #Find all 'images' directories
        images = os.listdir(root) 
        #print(images)
        for i, name in enumerate(images):  
            if name.endswith(".jpg"):  
               
                img = cv2.imread(root+"/"+name, 1)  #Read each image as BGR
                SIZE_X = (img.shape[1]//size)*size #Nearest size divisible by our patch size
                SIZE_Y = (img.shape[0]//size)*size #Nearest size divisible by our patch size
                img = Image.fromarray(img)
                img = img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                img = np.array(img)             
       
                #Extract patches from each image
                print("Patchifyed image:", root+"/"+name)
                patched_img = patchify(img, (size, size, 3), step=size)  #Step=256 for 256 patches means no overlap
                #print(patched_img.shape)
                for i in range(patched_img.shape[0]):
                    for j in range(patched_img.shape[1]):
                        
                        single_patch_img = patched_img[i,j,:,:]
                        
                        #Use minmaxscaler instead of just dividing by 255. 
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        #print(single_patch_img.shape)
                       
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                        #print(single_patch_img.shape)
                        processed_data.append(single_patch_img)
                

# Mask
processed_mask = []  
for root, subdirs, files in os.walk(root_dir): 
    dirname = root.split(os.path.sep)[-1]
    #print(dirname)
    if dirname == 'masks':   
        masks = os.listdir(root) 
        #print(masks)
        for i, name in enumerate(masks):  
            if name.endswith(".png"):  
               
                mask = cv2.imread(root+"/"+name, 1)  #Read each image as BGR
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (img.shape[1]//size)*size #Nearest size divisible by our patch size
                SIZE_Y = (img.shape[0]//size)*size #Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                mask = np.array(mask)             
       
                #Extract patches from each image
                print("Patchifyed mask:", root+"/"+name)
                patched_mask = patchify(mask, (size, size, 3), step=size)  #Step=256 for 256 patches means no overlap
                #print(patched_mask.shape)
                for i in range(patched_mask.shape[0]):
                    for j in range(patched_mask.shape[1]):
                        
                        single_patch_mask = patched_mask[i,j,:,:]
                        
                        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                        #print(single_patch_mask.shape)
                        processed_mask.append(single_patch_mask)


 
image_dataset = np.array(processed_data)
mask_dataset =  np.array(processed_mask)

#Sanity check
import random
import numpy as np

num= random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[num])
print('img shape:',image_dataset[num].shape)
plt.subplot(122)
plt.imshow(mask_dataset[num])
print('mask shape:',mask_dataset[num].shape)
plt.show()


a=int('3C', 16)  #3C with base 16. Should return 60. 
print(a)

# hex code converted to RGB
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

#label = single_patch_mask

def rgb_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg



labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)
 
print("Unique labels: ", np.unique(labels))

#Another Sanity check, view few mages
import random
import numpy as np
num = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[num])
plt.subplot(122)
plt.imshow(labels[num][:,:,0])
plt.show()

classes = len(np.unique(labels))
from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=classes)
'''
for i in range(len(image_dataset)):
    cv2.imwrite('./saved_data/images/'+'Image_{}.png'.format(i),(255*image_dataset[i]).astype(np.uint8))


for i in range(len(labels)):
    cv2.imwrite('./saved_data/masks/'+'Mask_{}.png'.format(i), labels[i])
    
'''
