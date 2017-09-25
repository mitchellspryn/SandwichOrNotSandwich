from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K

import numpy as np

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#TODO: For some reason, applying this preprocessing function to the VGG models seems to hurt accuracy.
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr (this also doesn't seem to work if it's 'return x[:, :, :-1]')

#The main factory method to be used by external callers to make models
def MakeModel(model_name):
    #All models take same input
    in_layer = Input(shape=(224,224,3))

    if model_name == 'simple':
        model = MakeSimpleModel(in_layer)
    elif model_name == 'vgg_finetune':
        model = MakeVggFinetuneModel(in_layer)
    elif model_name == 'vgg_finetune_bn':
        model = MakeVggFinetuneWithBnModel(in_layer)
    else:
        raise ValueError('Unrecognized model name: {0}'.format(model_name))

    model.compile(optimizer='adam', 
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model

#Makes a simple from-scratch model.
def MakeSimpleModel(in_layer):
    c1_1 = Conv2D(16, (3,3), activation='relu', padding='same')(in_layer)
    c1_2 = Conv2D(16, (3,3), activation='relu', padding='same')(c1_1)
    p1 = MaxPooling2D(pool_size=(2,2))(c1_2)

    c2_1 = Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    c2_2 = Conv2D(32, (3,3), activation='relu', padding='same')(c2_1)
    p2 = MaxPooling2D(pool_size=(2,2))(c2_2)
    
    f = Flatten()(p2)

    do3 = Dropout(0.2)(f)
    bn3 = BatchNormalization()(do3)
    de3 = Dense(32)(bn3)
    a3 = Activation('relu')(de3)

    do4 = Dropout(0.2)(a3)
    bn4 = BatchNormalization()(do4)
    de4 = Dense(2)(bn4)
    out = Activation('softmax')(de4)

    return Model(in_layer, out) 

#Makes the VGG16 model, freezes all layers, pops off the last layer, and uses it for retraining
def MakeVggFinetuneModel(in_layer):
    #x = Lambda(vgg_preprocess, input_shape=(224,224,3), output_shape=(224,224,3))(in_layer) #TODO: This should be a necessary preprocessing step. But it hurts performance.
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(in_layer) 
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    removeme = Dense(1000, activation='softmax')(x)
    
    # Create dummy model so we can use load_wights().
    model = Model(in_layer, removeme, name='dummy')
    
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
    
    #weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                WEIGHTS_PATH_NO_TOP,
    #                                cache_subdir='models')
    
    model.load_weights(weights_path)
    
    for layer in model.layers:
        layer.trainable = False
    
    #Finish up the model by adding our fine-tune layer onto the end and the preprocessing layer
    model.layers.pop()
    x = model.layers[-1].output
    out = Dense(2, activation='softmax', name='output')(x)
    model = Model(in_layer, out, name='vgg_finetune')
        
    return model
    
#VGG with inserted dropout and batchnormalization.
#As written, this doesn't give significant improvement over the standard fine-tuned VGG16 model.
#Excersize left to the reader to improve
def MakeVggFinetuneWithBnModel(in_layer):
    #x = Lambda(vgg_preprocess, input_shape=(224,224,3), output_shape=(224,224,3))(in_layer) #TODO: This should be a necessary preprocessing step. But it hurts performance.
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(in_layer) 
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Flatten(name='flatten')(x)
    #x = Dropout(0.3)(x)          #Will be added later
    #x = BatchNormalization()(x)  #Will be added later
    x = Dense(4096, activation='relu', name='fc1')(x)
    #x = Dropout(0.3)(x)          #Will be added later
    #x = BatchNormalization()(x)  #Will be added later
    x = Dense(4096, activation='relu', name='fc2')(x)
    #x = Dropout(0.3)(x)          #Will be added later
    #x = BatchNormalization()(x)  #Will be added later
    removeme = Dense(1000, activation='softmax')(x)
    
    # Create dummy model so we can use load_wights().
    model = Model(in_layer, removeme, name='dummy')
    
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
    
    model.load_weights(weights_path)
    
    #Remove the last layer
    model.layers.pop()
    
    #Set the pretrained layers to false.
    #Also grab references to a few key layers so we can insert the dropout and batchnormalization layers
    flat_layer = None
    first_dense = None
    second_dense = None
    for layer in model.layers:
        layer.trainable = False
        config = layer.get_config()
        if config['name'] == 'flatten':
            flat_layer = layer
        if config['name'] == 'fc1':
            first_dense = layer
        if config['name'] == 'fc2':
            second_dense = layer
    
    #Create and inject the dropout and batchnormalization layers
    do1 = Dropout(0.3)
    bn1 = BatchNormalization()
    do2 = Dropout(0.3)
    bn2 = BatchNormalization()
    do3 = Dropout(0.3)
    bn3 = BatchNormalization()
    out = Dense(2, activation='softmax', name='output')
    
    x = do1(flat_layer.output)
    x = bn1(x)
    x = first_dense(x)
    x = do2(x)
    x = bn2(x)
    x = second_dense(x)
    x = do3(x)
    x = bn3(x)
    x = out(x)
    
    model = Model(in_layer, x, name='vgg_finetune')
    
    #Uncomment the following lines to see which layers are trainable
    #for layer in model.layers:
    #    if ('trainable' in layer.get_config()):
    #        print('{0}|{1}'.format(layer.get_config()['name'], layer.get_config()['trainable']))
        
    return model
