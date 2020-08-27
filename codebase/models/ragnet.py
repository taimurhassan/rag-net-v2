import numpy as np
import keras
from keras.models import *
from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

#from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model, resize_image
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .rag_models import rag_encoder
from .dilated_encoder import get_dilated_encoder

def get_ragnet_classifier(encoder=get_dilated_encoder,  input_height=576, input_width=768, validate=False, model=None):
    assert input_height % 192 == 0
    assert input_width % 192 == 0
    assert model != None
	
    img_input, levels = encoder(input_height=input_height,  input_width=input_width)
    [f1, f2, f3, f4, f5] = levels
    #print(f5)
    #input_B_out_A = Input(shape=(12, 18, 256))
    # Concatenating the two input layers
    #concat = keras.layers.concatenate([f5, input_B_out_A])

    flat = Flatten()(f5)
    hidden1 = Dense(10, activation='relu')(flat)
    output = Dense(2, activation='softmax')(hidden1)  # Normal, Glaucoma
    
    t_height = input_width;
    t_width = input_height;
    nb_train_samples = 184;
    batch_size = 2;
    epochs = 40
    modelClassifier = Model(inputs=img_input, outputs=output)
    modelClassifier.model_name = "ragnet_classifier"
	#sgd = optimizers.SGD(lr=0.001, decay=0.5, momentum=0.9, nesterov=True)
    #modelClassifier.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  
    #transfer_weights(modelClassifier, model, verbose=True)  
    for i in range(1,len(modelClassifier.layers)-3):
        modelClassifier.layers[i].set_weights(model.layers[i].get_weights())
    #modelClassifier.summary()
    #sgd = optimizers.SGD(lr=0.001, decay=0.5, momentum=0.9, nesterov=True)
    modelClassifier.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])               
    train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    train_generator = train_datagen.flow_from_directory("C:/tbme/codebase/models/trainingSet", target_size=(t_width, t_height), class_mode='categorical', batch_size=batch_size)
    
    if validate:
        steps = 8 // batch_size
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_datagen.flow_from_directory("C:/tbme/codebase/models/trainingSet",(t_width, t_height),class_mode='categorical', batch_size=batch_size)
        modelClassifier.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=steps)        
    else:	
        modelClassifier.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs)
    
    modelClassifier.summary()
    return modelClassifier
	
def get_ragnet_segmentor(classes, encoder,  input_height=384, input_width=576):

    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)
    [f1, f2, f3, f4, f5] = levels
    #print(f5)
    o = decoder(f5, classes)
	
    modelSegmentor = get_segmentation_model(img_input, o)
    return modelSegmentor

def ragnet(n_classes,  input_height=384, input_width=576):

    modelSegmentor = get_ragnet_segmentor(n_classes, rag_encoder,
                    input_height=input_height, input_width=input_width)
    
    modelSegmentor.model_name = "ragnet"

    return modelSegmentor


def vgg_ragnet(n_classes,  input_height=384, input_width=576):

    modelSegmentor = get_ragnet_segmentor(n_classes, get_vgg_encoder,
                    input_height=input_height, input_width=input_width)
    
    modelSegmentor.model_name = "vgg_ragnet"
	
    return modelSegmentor

def dilated_ragnet(n_classes,  input_height=384, input_width=576):

    modelSegmentor = get_ragnet_segmentor(n_classes, get_dilated_encoder,
                    input_height=input_height, input_width=input_width)
    modelSegmentor.model_name = "dilated_ragnet"
	
    return modelSegmentor

def ragnet_50(n_classes,  input_height=473, input_width=473):
    from ._ragnet_2 import _build_ragnet

    nb_classes = n_classes
    resnet_layers = 50
    input_shape = (input_height, input_width)
    model = _build_ragnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "ragnet_50"
    return modelSegmenter


def ragnet_101(n_classes,  input_height=473, input_width=473):
    from ._ragnet_2 import _build_ragnet

    nb_classes = n_classes
    resnet_layers = 101
    input_shape = (input_height, input_width)
    model = _build_ragnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "ragnet_101"
    return model

def decoder(features, classes):
    from .config import IMAGE_ORDERING as order

    pool_factors = [1, 2, 8, 16]
    list = [features]
    print(len(list))
    
    if order == 'channels_first':
        h = K.int_shape(features)[2]
        w = K.int_shape(features)[3]
    elif order == 'channels_last':
        h = K.int_shape(features)[1]
        w = K.int_shape(features)[2]

    pool_size = strides = [
    int(np.round((float(h)+ 1)/3)),
    int(np.round(((float(w) + 1)/3)))]

    pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(features)
    pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
    pooledResult = BatchNormalization()(pooledResult)
    pooledResult = Activation('relu')(pooledResult)
        
    pooledResult = resize_image(pooledResult, strides, data_format=order)
    #list[0] = pooledResult
        
    for p in pool_factors:
        if order == 'channels_first':
            h = K.int_shape(features)[2]
            w = K.int_shape(features)[3]
        elif order == 'channels_last':
            h = K.int_shape(features)[1]
            w = K.int_shape(features)[2]

        pool_size = strides = [
            int(np.round((float(h))/ p)),
            int(np.round((float(w))/ p))]

        pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(features)
        pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
        pooledResult = BatchNormalization()(pooledResult)
        pooledResult = Activation('relu')(pooledResult)
        
        pooledResult = resize_image(pooledResult, strides, data_format=order)
        list.append(pooledResult)
		
    if order == 'channels_first':
        features = Concatenate(axis=1)(list)
    elif order == 'channels_last':
        features = Concatenate(axis=-1)(list)

    features = Conv2D(512, (1, 1), data_format=order, use_bias=False)(features)
    features = BatchNormalization()(features)
    features = Activation('relu')(features)

    features = Conv2D(classes, (3, 3), data_format=order,
               padding='same')(features)
    features = resize_image(features, (8, 8), data_format=order)

    return features


if __name__ == '__main__':

    m = _ragnet(101, rag_encoder)
    m = _ragnet(101, get_vgg_encoder)
    m = _ragnet(101, get_dilated_encoder)
