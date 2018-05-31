
# coding: utf-8

# In[1]:


import struct
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import backend as K
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
'''
from keras.datasets import mnist
from keras.layers import Input, Dropout,Lambda, Dense, Conv2D,Flatten,AveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D

import matplotlib.pyplot as plt
import datetime
K.set_image_dim_ordering('th')


# In[2]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# In[3]:


plt.imshow(X_train[0][0],cmap=plt.get_cmap('gray'))
plt.show()


# In[4]:


X_train = X_train /255
X_test = X_test /255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]


# In[84]:


from keras.utils.generic_utils import get_custom_objects
def custom_STF_activation(x,alpha=1.4):
    ''' 
       Arguments:
       x: input tensor
       alpha: index
       Returns:
       Tensor, output of stf function
    '''
    x_ = tf.nn.relu(x)
    y = x_**alpha
    return (y)

## how to define the log function
def custom_STD_activation(x,beta=1.1):
    x_ = tf.nn.relu(x)
    denominator = tf.log(tf.constant(beta,dtype=x_.dtype))
    y = x_/denominator
    return (y)


# def base_model():
#     model = Sequential()
#     model.add(Convolution2D(32,5,5,border_mode='valid',input_shape=(1,28,28),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128,activation='relu'))
#     model.add(Dense(num_class,activation='sigmoid'))
#     model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#     return model

# seed = 13
# np.random.seed(seed)
# model = base_model()
# model.fit(X_train,y_train,validation_data=(X_test,y_test),nb_epoch=10,batch_size=200,verbose=2)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("CNN Error: %.2f%%" % (100-scores[1]*100))

# In[85]:


def base_model2(lr=0.0005,decay=0.0):
    input_layer =Input(shape=(1,28,28))
    conv_layer1=Conv2D(6,5,padding='valid',activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_layer1)
    conv_layer2=Conv2D(16,5,padding='valid',activation=custom_STD_activation)(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_layer2)
    main2 = Dropout(0.2)(max_pool2)
    main2 = Flatten()(main2)
    main2 = Dense(128,activation='relu')(main2)
    main2 = Dense(84,activation='relu')(main2)
    output = Dense(10,activation='softmax')(main2)
    model = Model(input_layer,output)
   # optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model


# In[86]:


model = base_model2()
model.summary()
del model


# In[87]:


# Set hyper parameters for the model.
BATCH_SIZE = 256
epochs = 10
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)
cnn_model2 = base_model2()
cnn_model2.fit(x=X_train,y=y_train,batch_size=16,epochs=20,verbose=1,validation_data=(X_test,y_test))


# In[17]:


fig, ax = plt.subplots(2,1)
ax[0].plot(cnn_model2.history.history['loss'], color='b', label="Training loss")
ax[0].plot(cnn_model2.history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(cnn_model2.history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(cnn_model2.history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()


# In[5]:


def base_model2_adam(lr=0.0005,decay=0.0):
    input_layer =Input(shape=(1,28,28))
    conv_layer1=Conv2D(6,5,padding='valid',activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_layer1)
    conv_layer2=Conv2D(16,5,padding='valid',activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_layer2)
    main2 = Dropout(0.2)(max_pool2)
    main2 = Flatten()(main2)
    main2 = Dense(128,activation='relu')(main2)
    main2 = Dense(84,activation='relu')(main2)
    output = Dense(10,activation='softmax')(main2)
    model = Model(input_layer,output)
    optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.89,epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
model = base_model2_adam()
model.summary()
del model


# In[6]:


# Set hyper parameters for the model.
datagen = ImageDataGenerator(zoom_range = 0.05,
                            height_shift_range = 0.05,
                            width_shift_range = 0.05,
                            rotation_range = 3)
BATCH_SIZE = 256
epochs = 10
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)
cnn_model3 = base_model2_adam()
cnn_model3.fit_generator(
        datagen.flow(X_train, y_train, batch_size=256),steps_per_epoch=1000, epochs=epochs,
        validation_data=(X_test, y_test), verbose=1,
)


# In[11]:


def base_model3(lr=0.001,decay=0.0):
    input_layer =Input(shape=(1,28,28))
    conv_layer1=Conv2D(30,(5,5),padding='valid',activation='relu')(input_layer)
    avg_pool1 = MaxPooling2D(pool_size=(2,2),strides=1)(conv_layer1)
    conv_layer2=Conv2D(15,(3,3),padding='valid',activation='relu')(avg_pool1)
    avg_pool2 = MaxPooling2D(pool_size=(2,2),strides=1)(conv_layer2)
    main2 = Dropout(0.2)(avg_pool2)
    main2 = Flatten()(main2)
    main2 = Dense(128,activation='relu')(main2)
    #main2 = Dropout(0.1)(Dense(128,activation='relu')(main2))
    main2 = Dense(84,activation='relu')(main2)
    output = Dense(10,activation='softmax')(main2)
    model = Model(input_layer,output)
    optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.9,epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
model = base_model3()
model.summary()
del model


# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.05,
                            height_shift_range = 0.05,
                            width_shift_range = 0.05,
                            rotation_range = 3)
BATCH_SIZE = 256
epochs = 10
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.007, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)
cnn_model = base_model3(lr=lr_init,decay=0)
cnn_model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=256),steps_per_epoch=1000, epochs=epochs,
        validation_data=(X_test, y_test), verbose=1,
)


# In[45]:


y_test_predict = cnn_model.predict(X_test)


# def larger_model():
# # create model
#     model = Sequential()
#     model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Convolution2D(15, 3, 3, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# # build the model
# model = larger_model()
# # Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Larger CNN Error: %.2f%%" % (100-scores[1]*100))

# In[15]:


def base_model4(lr=0.0005,decay=0.0):
    input_layer =Input(shape=(1,28,28))
    conv_layer1 = Conv2D(32,(5,5),padding = 'same',activation='relu')(input_layer)
    conv_layer2 = Conv2D(32,(5,5),padding = 'same',activation='relu')(conv_layer1)
    max_pool1 = MaxPooling2D(pool_size=(2,2))(conv_layer2)
    main1 = Dropout(0.25)(max_pool1)
    conv_layer3 = Conv2D(64,(3,3),padding='same',activation='relu')(main1)
    conv_layer4 = Conv2D(64,(3,3),padding='same',activation='relu')(conv_layer3)
    max_pool2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_layer4)
    main2 = Dropout(0.2)(max_pool2)
    main2 = Flatten()(main2)
    main2 = Dense(256,activation='relu')(main2)
    main2 = Dense(128,activation='relu')(main2)
    output = Dense(10,activation='softmax')(main2)
    model = Model(input_layer,output)
    #optimizer = Adam(lr=lr,decay=decay)
    optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
    
model = base_model4()
model.summary()
del model


# In[18]:


datagen = ImageDataGenerator(zoom_range = 0.05,
                            height_shift_range = 0.05,
                            width_shift_range = 0.05,
                            rotation_range = 3)
BATCH_SIZE = 200
epochs = 20
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.002, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)
cnn_model = base_model4(lr=lr_init,decay=0)
cnn_model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=256),steps_per_epoch=1000, epochs=epochs,
        validation_data=(X_test, y_test), verbose=1,
)


# In[38]:


def base_model4_adam(lr=0.0005,decay=0.0):
    input_layer =Input(shape=(1,28,28))
    conv_layer1 = Conv2D(32,(5,5),padding = 'same',activation='relu')(input_layer)
    conv_layer2 = Conv2D(32,(5,5),padding = 'same',activation='relu')(conv_layer1)
    max_pool1 = MaxPooling2D(pool_size=(2,2))(conv_layer2)
    main1 = Dropout(0.25)(max_pool1)
    conv_layer3 = Conv2D(64,(3,3),padding='same',activation='relu')(main1)
    conv_layer4 = Conv2D(64,(3,3),padding='same',activation='relu')(conv_layer3)
    max_pool2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_layer4)
    main2 = Dropout(0.2)(max_pool2)
    main2 = Flatten()(main2)
    main2 = Dense(256,activation='relu')(main2)
    main2 = Dense(128,activation='relu')(main2)
    output = Dense(10,activation='softmax')(main2)
    model = Model(input_layer,output)
    #optimizer = Adam(lr=lr,decay=decay)
    optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.9,epsilon=1e-08,decay=0.0)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
    
model = base_model4()
model.summary()
del model


# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.05,
                            height_shift_range = 0.05,
                            width_shift_range = 0.05,
                            rotation_range = 3)
BATCH_SIZE = 200
epochs = 5
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.002, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)
cnn_model = base_model4(lr=lr_init,decay=0)
cnn_model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=256),steps_per_epoch=1000, epochs=epochs,
        validation_data=(X_test, y_test), verbose=1,
)

