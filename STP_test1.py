
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
import itertools
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


# ## define the STD and STF activation function

# In[5]:


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


# ## Preprocessing the data
#     1. extract digit 1% digit 5 from its original dataset to create imbalance data
#        further study: we can extract other to create different to test
#     2. whether the different digit will change the result

# In[6]:


def create_imbalance_data(x_train,y_train,x_test,y_test,target_digit=1):
    x_train_tmp = x_train
    #choose the digit that is not the target
    x_not_extract = x_train[y_train!=target_digit]
    y_not_extract = y_train[y_train!=target_digit]
    x_extract = x_train_tmp[y_train==target_digit]
    y_extract = y_train[y_train==target_digit]
    x_train_imbalance, x_test_imbalance, y_train_imbalance, y_test_imbalance = train_test_split(x_extract,y_extract,
                                                                                            test_size=0.99,random_state=42)
    X_train = np.concatenate((x_not_extract,x_train_imbalance))
    y_train = np.concatenate((y_not_extract,y_train_imbalance))
    return X_train,y_train


# In[7]:


X_train,y_train = create_imbalance_data(X_train,y_train,X_test,y_test,target_digit=5)


# In[8]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]
print("number of class is: %d"%(num_class))


# In[9]:


indices = np.random.permutation(X_train.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量
rand_data_x = X_train[indices]
rand_data_y = y_train[indices] # data_y就是标记（label）


# In[10]:


rand_data_x.shape


# In[11]:


plt.imshow(rand_data_x[26][0],cmap=plt.get_cmap('gray'))
plt.show()


# In[12]:


rand_data_y[26]


# In[13]:


X_train = rand_data_x
y_train = rand_data_y


# In[14]:


X_test.shape


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

# In[15]:


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


# In[16]:


model = base_model2()
model.summary()
del model


# In[17]:


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


# In[18]:


fig, ax = plt.subplots(2,1)
ax[0].plot(cnn_model2.history.history['loss'], color='b', label="Training loss")
ax[0].plot(cnn_model2.history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(cnn_model2.history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(cnn_model2.history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()


# In[19]:


def evaluate(prediction, true_labels):
    pred = np.argmax(prediction, 1)
    true = np.argmax(true_labels, 1)

    equal = (pred == true)

    return np.mean(equal)


# In[20]:


def get_cm(y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    return cm


# In[21]:


def plot_confusion_matrix(predict_label,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,acc=0.0):
    cm = get_cm(predict_label)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix %f'%(acc))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[22]:


final_pred = cnn_model2.predict(X_test)
acc = evaluate(final_pred, y_test)


# In[23]:


plot_confusion_matrix(final_pred,classes = range(10),acc=acc) 


# In[26]:


def base_model2_relu(lr=0.0005,decay=0.0):
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
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
model = base_model2_relu()
model.summary()
del model


# In[27]:


# Set hyper parameters for the model.
BATCH_SIZE = 256
epochs = 10
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)
cnn_model_relu = base_model2_relu()
cnn_model_relu.fit(x=X_train,y=y_train,batch_size=16,epochs=20,verbose=1,validation_data=(X_test,y_test))


# In[28]:


final_pred = cnn_model_relu.predict(X_test)
acc = evaluate(final_pred, y_test)
plot_confusion_matrix(final_pred,classes = range(10),acc=acc) 

