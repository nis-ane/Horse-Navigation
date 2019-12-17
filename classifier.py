import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import sklearn
from sklearn.metrics import accuracy_score
from keras.models import load_model

#from keras.models import Sequential
#from keras.layers import *
#from keras.optimizers import *
import keras_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU

train_data = 'navigation_data/train'
test_data = 'navigation_data/test'
predict_data = 'navigation_data/predict_simulation'

def one_hot_label(img):                 #function for generating label for data
    label = img.split('.')[0]
    if label == 'zoneA':
        ohl = np.array([1,0,0])
    elif label == 'zoneB':
        ohl = np.array([0,1,0])
    elif label == 'zoneC':
        ohl = np.array([0,0,1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)           #path to the image in folder
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64,64))
        train_images.append([np.array(img), one_hot_label(i)])   #appended([image1],[hotlabel1],[image2],[hotlabel2])
    shuffle(train_images)    
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64,64))
        test_images.append([np.array(img), one_hot_label(i)])
    return test_images


def predict_data_with_label():
    predict_images = []
    for i in tqdm(os.listdir(predict_data)):
        path = os.path.join(predict_data, i)           #path to the image in folder
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64,64))
        predict_images.append([np.array(img)])   #appended([image1],[hotlabel1],[image2],[hotlabel2])   
    return predict_images


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val)/(max_val-min_val)
    return x



training_images = train_data_with_label()
testing_images = test_data_with_label() 
predicting_images = predict_data_with_label()    
#testing_images = [[array([[[51, 78, 94],
#                           [53, 77, 94],
#                           .........
#                           [100,105,103],
#                           [53,75,101]],
#                           
#                           ........
#            
#                           [[51, 78, 94],
#                           [53, 77, 94],   
#                           ..........                           
#                           [ 53,  77,  96],
#                           [ 48,  79,  94]]], dtype=uint8), array([0, 1, 0])], ......]


print(len(training_images))
print(len(testing_images))
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,3)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,64,64,3)    
tst_lbl_data = np.array([i[1] for i in testing_images])

prd_img_data =np.array([i[0] for i in predicting_images]).reshape(-1,64,64,3)
#tst_lbl_data = #[[[[51  77  96]
#                   [51 77 96]
#                   .......
#                   [53 77 96]
#                   [54 77 96]]]]
#                

print(tr_img_data.shape)
print(tr_lbl_data.shape)
print(tst_img_data.shape)
print(tst_lbl_data.shape)


tr_img_data2 = normalize(tr_img_data) 
tst_img_data2 = normalize(tst_img_data) 



#show random images from train
#cols = 8
#rows = 2
#fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
#for i in range(cols):
#    for j in range(rows):
#        random_index = np.random.randint(0, len(tr_lbl_data))
#        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
#        ax.grid('off')
#        ax.axis('off')
#        ax.imshow(tr_img_data2[random_index, :])
#plt.show()


def make_model():
    """
    Define your model architecture here.
    Returns `Sequential` model.
    """
    model = Sequential()

    ### YOUR CODE HERE
    model.add(Conv2D(16,(3,3),padding="same",input_shape=(64, 64, 3)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    model.add(Dropout(0.25))
    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(LeakyReLU(0.1))
    model.add(Activation("softmax"))
    return model

tf.reset_default_graph()  # clear default graph
model = make_model()
model.summary()


INIT_LR = 4e-3  # initial learning rate
BATCH_SIZE = 30
EPOCHS = 8

tf.reset_default_graph()  # clear default graph

model = make_model()  # define our model


# prepare model for fitting 
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))

model_filename = 'weights.hdf5'
last_finished_epoch = None

def plot_history(history):
    fig = plt.figure(figsize=(15, 7))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


history = model.fit(
    tr_img_data2, tr_lbl_data,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
               LrHistory()],
    validation_data=(tst_img_data2, tst_lbl_data),
    shuffle=True,
    verbose=0,
    initial_epoch=last_finished_epoch or 0
)

plot_history(history)

scores= model.evaluate(tr_img_data2, tr_lbl_data, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save_weights("weights.h5", overwrite= True)

model.load_weights("weights.h5")
y_pred_test = model.predict(prd_img_data)
print(y_pred_test)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
print(y_pred_test_classes)


