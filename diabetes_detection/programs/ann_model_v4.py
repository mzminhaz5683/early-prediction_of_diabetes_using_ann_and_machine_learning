#Module importingg
import keras, os
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import (Dense, Dropout, Activation, Flatten, LeakyReLU, BatchNormalization)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from programs import controler
from programs import model_database

from programs.checker_v2 import accuracy_calculator
from programs.controler import initial_weight
from programs.controler import alpha_lrelu
from programs.controler import leakyRelu
from programs.controler import activate_train
from programs.controler import test_parameters
from programs.controler import dropout
from programs.controler import dns
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
####################################################################################################
#                                   Load project
####################################################################################################
output_path = './output/predicted_results/'
project = ''
random_split = 1
if controler.project_version == 3:
        from programs import project_v3_actual_split as project_analyser
        project = 'project_v3_actual_split'
        random_split = 0
elif controler.project_version == 5:
        from programs import project_v5_random_split as project_analyser
        project = 'project_v5_random_split'
else:
    print("\n\n\n Can't find any process model \n\n\n")
    exit(0)
####################################################################################################
#                                   Load documents
####################################################################################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('             model start for : {0}'.format(project))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
try:
        y_test = project_analyser.get_actual_result()

except:
        actual_result = pd.read_csv("./input/actual_result.csv")
        y_test = actual_result['Outcome'] # actual result

X_train, X_test = project_analyser.get_train_test_data()
y_train = project_analyser.get_train_label()
X_test_ID, X_train_ID = project_analyser.get_IDs()
####################################################################################################
#                                   ANN parameters
####################################################################################################
random_train = pd.concat([X_train,X_test])
random_y     = pd.concat([y_train, y_test])

# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
ann_train, ann_test, ann_y_train, ann_y_test = train_test_split(random_train, random_y, test_size=0.15) # 80% training and 20% test

print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
print('~~~~~~~~~ Random_shape for ANN training ~~~~~~~~~')
print('\nann~train, ann~y~train :{0}, {1}\n\nann~test, ann~y~test :{2}, {3}'.format(
                ann_train.shape, ann_y_train.shape, ann_test.shape, ann_y_test.shape))
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')

regularizer = keras.regularizers.l2(initial_weight)

####################################################################################################
#                                   ANN Model
####################################################################################################
def get_model():
        #Single layer architecture
        model = Sequential()

        model.add(Dense(dns[0],input_shape=(ann_train.shape[1],), kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Dense(dns[1]))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate = dropout[0]))
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Dense(dns[2]))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Dense(dns[3]))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate = dropout[1]))
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Dense(dns[4]))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Dense(dns[5]))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Dense(dns[6]))
        model.add(LeakyReLU(alpha=alpha_lrelu)) if (leakyRelu) else model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate = dropout[2]))
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------
        model.add(Flatten())

        model.add(Dense(2))
        model.add(Activation('softmax'))

        # print model
        model.summary()
        return model

####################################################################################################
#                                   model function
####################################################################################################
model_checkpoint_dir = './output/checkpoint/ann_model.h5'
saved_model_dir = './output/checkpoint/saved/{0}.h5'.format(test_parameters)
monitor = 'val_loss'
#read mode
def read_model(model_parameters):
    model = load_model(model_parameters)
    return model

#save check-point model
def saved_model_checkpoint():
    return ModelCheckpoint(model_checkpoint_dir, 
                            monitor=monitor, 
                            verbose=2, 
                            save_best_only=True, 
                            save_weights_only=False, 
                            mode='auto', 
                            period=1)

#early stopping of model
def set_early_stopping():
    return EarlyStopping(monitor=monitor,
                            patience= 15,
                            verbose=2,
                            mode='auto')

model_optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
def reduce_lr():
    print("Optimizer :","Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)")
    return ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience=5,
                            min_lr=0.001)
####################################################################################################
#                                   training
####################################################################################################
if activate_train:   
        model = get_model()
        model_cp = saved_model_checkpoint()
        lr_controller = reduce_lr()
        early_stopping = set_early_stopping()
        ####################################################################################################

        model.compile(loss='sparse_categorical_crossentropy',
                optimizer="sgd",metrics=['accuracy'])

        if 0:
                model.fit(ann_train, ann_y_train,
                        epochs=100,
                        verbose=2, # 1 = 71
                        validation_data=(ann_test, ann_y_test),
                        callbacks=[early_stopping, model_cp, lr_controller])
        else:
                model.fit(ann_train, ann_y_train,
                epochs=100,
                verbose=2, # 1 = 71
                validation_data=(ann_test, ann_y_test),
                callbacks=[model_cp, lr_controller])  

        loss, accuracy = model.evaluate(ann_test, ann_y_test, verbose=0)

        print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print("\nANN Loss : "+str(loss))
        print("\n\nANN Accuracy : {0:.2f} %".format(accuracy*100.0))
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
####################################################################################################
#                                   predecting
####################################################################################################
if activate_train:
        test_model = read_model(model_checkpoint_dir)
else:
        test_model = read_model(saved_model_dir)

predictions = test_model.predict(X_test)
label_pred = np.argmax(predictions, axis = 1)

#convet numpy vector into list
result = []
result += label_pred.tolist()

y_pred_f, acc = accuracy_calculator('\n\nANN', result, y_test)

if acc > 85:
        destination = './output/set_of_+80_ann_h5/{0:.1f}_ann_model.h5'.format(acc)
        string = "cp '{0}' '{1}'".format(model_checkpoint_dir, destination)
        #print(string)
        os.popen(string)
