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

initial_weight = 0.01
alpha_lrelu = 0 # 0/0.1
leakyRelu = 1 #0/1

dropout = [0.10, 0.15, 0.20]
#dns = [32, 32, 64, 128, 128, 256]
dns = [32, 64, 64, 128, 128, 128, 256]
regularizer = keras.regularizers.l2(initial_weight)

####################################################################################################
#                                   ANN Model
####################################################################################################
def get_model():
        #Single layer architecture
        model = Sequential()

        model.add(Dense(dns[0],input_shape=(X_train.shape[1],), kernel_regularizer=regularizer))
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
saved_model_dir = './output/checkpoint/save/ann_model.h5'
monitor = 'val_loss'

#read mode
def read_model():
    model = load_model(saved_model_dir)
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
#                                   load model for train, test
####################################################################################################
model = get_model()
model_cp = saved_model_checkpoint()
lr_controller = reduce_lr()
early_stopping = set_early_stopping()
####################################################################################################
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="sgd",metrics=['accuracy'])

if 0:
        model.fit(X_train, y_train,
                epochs=100,
                verbose=2, # 1 = 71
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, model_cp, lr_controller])
else:
        model.fit(X_train, y_train,
        epochs=100,
        verbose=2, # 1 = 71
        validation_data=(X_test, y_test),
        callbacks=[model_cp, lr_controller])  

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

predictions = model.predict(X_test)
label_pred = np.argmax(predictions, axis = 1)

#convet numpy vector into list
result = []
result += label_pred.tolist()

if 1:
        y_pred_f, acc = accuracy_calculator('ANN', result, y_test)
        print("\n\ntest Accuracy : {0:.2f} %".format(acc))


print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("\nANN Loss : "+str(loss))
print("\n\nANN Accuracy : {0:.2f} %".format(accuracy*100.0))
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
