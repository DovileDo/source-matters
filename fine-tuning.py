#!/usr/bin/env python
# coding: utf-8


import argparse
parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, help='choose RadImageNet or ImageNet')
parser.add_argument('--target', type=str, help='choose isic, chest, pcam-middle, thyroid, breast')
parser.add_argument('--dir', type=str, help='training data directory')
parser.add_argument('--k', type=int, help='fold num', default=1)
parser.add_argument('--batch_size', type=int, help='batch size', default=256)
parser.add_argument('--image_height', type=int, help='image height')
parser.add_argument('--image_width', type=int, help='image width')
parser.add_argument('--epoch', type=int, help='number of epochs', default=30)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--multiprocess', type=bool, help='use multiprocess', default=False)
args = parser.parse_args()

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras import regularizers, activations
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from PIL import ImageFile
from sklearn.preprocessing import OneHotEncoder
import radt
from io_functions.data_paths import get_path

keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

gpus = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

### Import pre-trained weights from ImageNet or RadImageNet
base = args.base
if not base in ['RadImageNet', 'ImageNet', 'random', 'chest14']:
    raise Exception('Pre-trained database not exists. Please choose ImageNet or RadImageNet')

target = args.target
if not target in ['isic', 'chest', 'chest14', 'kimia', 'pcam-small', 'pcam', 'pcam-middle', 'rad_thyroid', 'thyroid', 'breast', 'knee', 'mammograms']:
    raise Exception('Target dataset not selected. Please choose isic, chest, pcam-middle/small, thyroid or breast')

if args.image_height is None:
    raise Exception('Image height not specified')

if args.image_width is None:
    raise Exception('Image width not specified')


### Set up training image size, batch size and number of epochs and home
image_height = args.image_height
image_width = args.image_width

train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                          rescale=1./255 if base == 'RadImageNet' else None,
                                          rotation_range=10,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          shear_range=0.1,
                                          zoom_range=0.1,
                                          fill_mode='nearest',
                                          horizontal_flip=False #if target == 'chest' else True,
                                          )

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255 if base == 'RadImageNet' else None)
if args.target == 'mammograms':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
# Load data
df_train = pd.read_csv("/home/doju/data/" + target + "/train_fold" + str(args.k) + ".csv")
df_val = pd.read_csv("/home/doju/data/" + target + "/val_fold" + str(args.k) + ".csv")
df_test = pd.read_csv("/home/doju/data/" + target + "/test.csv")

data_path = get_path('/home/data_shares/purrlab', args.target)
print(data_path+'/'+args.dir)

train_generator = train_data_generator.flow_from_dataframe(dataframe=df_train,
                                                            directory=data_path+'/'+args.dir,
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            class_mode='categorical')

validation_generator = data_generator.flow_from_dataframe(dataframe=df_val,
                                                            directory=data_path+'/'+args.dir,
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_val),
                                                            shuffle=True,
                                                            class_mode='categorical')

test_generator = data_generator.flow_from_dataframe(dataframe=df_test,
                                                            directory=data_path+'/'+args.dir,
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_test),
                                                            shuffle=False,
                                                            class_mdoe='categorical')

test_clean = data_generator.flow_from_dataframe(dataframe=df_test,
                                                            directory=data_path+'/images',
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_test),
                                                            shuffle=False,
                                                            class_mdoe='categorical')


test_R = data_generator.flow_from_dataframe(dataframe=df_test,
                                                            directory=data_path+'/test_R',
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_test),
                                                            shuffle=False,
                                                            class_mdoe='categorical')

test_low = data_generator.flow_from_dataframe(dataframe=df_test,
                                                            directory=data_path+'/test_low',
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_test),
                                                            shuffle=False,
                                                            class_mdoe='categorical')

test_noise = data_generator.flow_from_dataframe(dataframe=df_test,
                                                            directory=data_path+'/test_noise',
                                                            x_col = 'path',
                                                            y_col = 'class',
                                                            target_size=(image_height, image_width),
                                                            batch_size=len(df_test),
                                                            shuffle=False,
                                                            class_mdoe='categorical')



if base == 'random':
    base_model = ResNet50(weights=None, input_shape=(image_height, image_width, 3), include_top=False,pooling='avg')

elif base == 'RadImageNet':
     model_dir = "/home/doju/models/RadImageNet-ResNet50_notop.h5"
     base_model = ResNet50(weights=model_dir, input_shape=(image_height, image_width, 3), include_top=False,pooling='avg')
elif base == 'ImageNet':
     base_model = ResNet50(weights='imagenet', input_shape=(image_height, image_width, 3), include_top=False,pooling='avg')

y = base_model.output
y = Dropout(0.5)(y)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(y)
model = Model(inputs=base_model.input, outputs=predictions)

filepath= "/home/doju/models/" + target + "-" + base + "-" + str(args.dir) + "-fold" + str(args.k) + ".h5"  
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', patience=30)
#es = EarlyStopping(monitor='val_auc', patience=30)

train_steps =  len(train_generator.labels)/ args.batch_size
val_steps = len(validation_generator.labels) / args.batch_size

with radt.run.RADTBenchmark() as run:
    
    # Log parameters to mlflow
    for key, value in vars(args).items():
        run.log_param(key, value)

    for _ in range(1):
        image, label = train_generator.next()
        RGBimage = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        plt.imshow(RGBimage)
        plt.title('Label: {}'.format(label[0]))
        plt.axis('off')
        plt.savefig('train_example.png')
        run.log_artifact('train_example.png')
    
    ml_metrics = {}
    run.autolog(log_models=False)
    loss = CategoricalCrossentropy()
    adam = Adam(learning_rate=args.lr)
    model.compile(optimizer=adam, loss=loss, metrics=[keras.metrics.AUC(name='auc')])
    history = model.fit(train_generator,
                                 epochs=args.epoch,
                                 steps_per_epoch=train_steps,
                                 validation_data=validation_generator,
                                 callbacks=[es,checkpoint])
    
    FT_model = load_model(filepath)
    
    def get_auc(data_generator):
        predictions = FT_model.predict(data_generator) # get predictions
        onehot = OneHotEncoder()
        y = data_generator.classes
        yonehot = onehot.fit_transform(np.array(y).reshape(-1,1)).toarray()
        m = keras.metrics.AUC(name='auc')
        m.update_state(yonehot, predictions)
        auc = m.result().numpy()
        return auc
    
    ml_metrics['AUC'] = get_auc(test_generator)
    ml_metrics['AUC_R_negative'] = get_auc(test_R)
    ml_metrics['AUC_low_negative'] = get_auc(test_low)
    ml_metrics['AUC_noise_negative'] = get_auc(test_noise)
    ml_metrics['AUC_clean'] = get_auc(test_clean)
    run.log_metrics(ml_metrics, 0)
    run.log_artifact("/home/doju/models/" + target + "-" + base + "-" + (args.dir) + "-fold" + str(args.k) + ".h5")
