#!/usr/bin/env python3

'''
This code is used to loop over many training instances to train mutliple, independent betting models. 
'''

###################################################################################################################
# IMPORTS
###################################################################################################################

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Lambda, Conv1D, BatchNormalization

from tensorflow.compat.v1.keras.backend import set_session


###################################################################################################################
# GPU PARAMETERS
###################################################################################################################

# Select gpu to use (old=1, new=0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()

if os.environ["CUDA_VISIBLE_DEVICES"]=="1":
    config.gpu_options.per_process_gpu_memory_fraction = 0.30
else: 
    config.gpu_options.per_process_gpu_memory_fraction = 0.30
set_session(tf.compat.v1.Session(config=config))


###################################################################################################################
# LOAD DATA
###################################################################################################################

save_path = "data/validation_data/"
X_train = np.load(save_path + "X_train.npy")
Y_train = np.load(save_path + "Y_train.npy")
X_test = np.load(save_path + "X_test.npy")
Y_test = np.load(save_path + "Y_test.npy")
X_val = np.load(save_path + "X_val.npy")
Y_val = np.load(save_path + "Y_val.npy")


###################################################################################################################
# LOAD WIN-LOSS MODELS
###################################################################################################################

n_wl_models = 9
wl_models = [load_model(f"saved_models/win_loss_model/best_model_{i}") for i in range(n_wl_models)]


###################################################################################################################
# BUILD WIN-LOSS AVERAGE MODEL
###################################################################################################################

wl_input = Input(shape=(X_train.shape[1]))
wl_outputs = [m(wl_input) for m in wl_models]
avg_wl_output = tf.keras.layers.Average()(wl_outputs)

avg_wl_model = Model(inputs=wl_input, outputs=avg_wl_output)
avg_wl_model.trainable = False


###################################################################################################################
# DEFINE BETTING MODEL ARCHITECTURE
###################################################################################################################

regularizer = "l1"
def make_betting_model(data_input_tuple):
    
    # Make the model
    main_input = Input(shape=(data_input_tuple.shape[1]), name="main_input")
    odds_input = Input(shape=(2), name="odds_input")
    
    odds_concat = Concatenate()([main_input, odds_input])
    
    # Define the main branch
    hidden_layer = Dense(10, activation="relu", kernel_regularizer=regularizer)(odds_concat)
    hidden_output = Dense(2, activation="sigmoid")(hidden_layer)
    
    # Define the wl branch
    wl_output = avg_wl_model(main_input)
    
    # Append output to averaging model
    concat_layer = Concatenate()([hidden_output, wl_output])
    hidden_layer_2 = Dense(2, activation="relu", kernel_regularizer=regularizer)(concat_layer)
    output_layer = Dense(2, activation="sigmoid")(hidden_layer_2)
    
    model = Model(inputs=[main_input, odds_input], outputs=output_layer)
    
    
    return model
    

###################################################################################################################
# DEFINE TRAINING CALLBACKS
###################################################################################################################

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)]


###################################################################################################################
# DEFINE TRAINING PARAMETERS
###################################################################################################################

batch_size = 32

epochs = 700
learning_rate = 1e-4
learning_rate_decay = 0.00001

loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
metrics = ['accuracy']

n_models = 30

acc_thresh = 0.55


###################################################################################################################
# TRAIN MULTIPLE MODELS
###################################################################################################################

trained_models = []
for i in range(n_models):
    
    # Create model template
    model = make_wl_model(input_tuple=(X_train.shape[1],))

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    # Train the model
    model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), 
              batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
    
    # Check that it is better than the threshold
    val_acc = model.evaluate(X_val, Y_val, return_dict=True, verbose=0)['accuracy']
    
    print(f"{i}: Validation Accuracy: {val_acc*100:.2f}%")
    if val_acc > 0.55:
        trained_models.append(model)


###################################################################################################################
# SAVE THE MODELS
###################################################################################################################

for i, model in enumerate(models):
    model.save(f"saved_models/win_loss_model/best_model_{i}")