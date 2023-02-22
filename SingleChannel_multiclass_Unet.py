from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


def Unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_classes):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Encoder
    E1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    E1 = Dropout(0.2)(E1)
    E1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(E1)
    P1 = MaxPooling2D((2, 2))(E1)
    
    E2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P1)
    E2 = Dropout(0.2)(E2) 
    E2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(E2)
    P2 = MaxPooling2D((2, 2))(E2)
    
    E3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P2)
    E3 = Dropout(0.2)(E3)
    E3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(E3)
    P3 = MaxPooling2D((2, 2))(E3)
    
    E4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P3)
    E4 = Dropout(0.2)(E4)
    E4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(E4)
    P4 = MaxPooling2D(pool_size=(2, 2))(E4)
    
    E5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P4)
    E5 = Dropout(0.3)(E5)
    E5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(E5)
    
    # Decoder
    U4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(E5)
    U4 = concatenate([U4, E4])
    D4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U4)
    D4 = Dropout(0.2)(D4)
    D4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D4)
    
    U3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(D4)
    U3 = concatenate([U3, E3])
    D3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U3)
    D3 = Dropout(0.2)(D3)
    D3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D3)
    
    U2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(D3)
    U2 = concatenate([U2, E2])
    D2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U2)
    D2 = Dropout(0.2)(D2) 
    D2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D2)
    
    U1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(D2)
    U1 = concatenate([U1, E1], axis=3)
    D1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U1)
    D1 = Dropout(0.2)(D1)  
    D1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D1)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(D1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
