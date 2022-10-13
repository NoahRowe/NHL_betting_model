#!/usr/bin/env python3

###########################################################################################
# IMPORTS
###########################################################################################

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from create_architecture import make_dense_model

import sys
sys.path.append("../")
from tensorflow_gpu.configure import configure_tensorflow
configure_tensorflow(eager=False, gpu_index=1, allow_growth_gpu=False)


###########################################################################################
# LOAD IN THE TRAINING DATA
###########################################################################################

lookback_games = 13

data_path = "../data/"
X = np.load(data_path + f"double_mean_lg{lookback_games}_X.npy")
Y = np.load(data_path + f"double_mean_lg{lookback_games}_Y.npy")

# Split and scale the data
train_pct, test_pct, val_pct = 0.6, 0.2, 0.2

X_train = X[:int(train_pct*X.shape[0])]
X_test = X[int(train_pct*X.shape[0]):int((train_pct+test_pct)*X.shape[0])]
X_val = X[-int(val_pct*X.shape[0]):]

Y_train = Y[:int(train_pct*X.shape[0])]
Y_test = Y[int(train_pct*X.shape[0]):int((train_pct+test_pct)*X.shape[0])]
Y_val = Y[-int(val_pct*X.shape[0]):]

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


###########################################################################################
# DEFINE LOOP PARAMETERS 
###########################################################################################

hidden_layer_options = [
    [],
    [5],
    [10], 
    [20],
    [10, 5],
    [20, 5],
]

regularization_options = [
    None,
    tf.keras.regularizers.L1(0.01), # default params
    tf.keras.regularizers.L2(0.01), # default params
    tf.keras.regularizers.L1L2(l1=0.01, l2=0.01) # default params
]

n_models = 11
learning_rate = 2e-4
epochs = 300


#########################################################################################################
# MODEL EVALUATION FUNCTIONS
#########################################################################################################

def eval_models(hidden_layer_sizes, regularization):
    
    # Loop through and train models
    trained_models = []
    for _ in range(n_models):
        
        # Define the model structure
        model = make_dense_model(input_size=X.shape[1], 
                                 hidden_layer_sizes=hidden_layer_sizes, 
                                 regularization=regularization, activation="relu")
        
        # Compile the model
        loss = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        metrics = ['accuracy']
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        # Fit the model
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
        model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), 
                  batch_size=64, epochs=epochs, callbacks=callbacks, verbose=0)
        
        trained_models.append(model)
        
    # Evaluate all the models together
    final_accuracy = evaluate_double_models(trained_models, X_val, Y_val)
        
    print("\nModel:", str(hidden_layer_sizes), str(regularization))
    print(f"Final Accuracy: {final_accuracy*100:.2f}%.")
    return final_accuracy

def evaluate_double_models(models, X, Y):
    
    # Generate predictions
    predictions = np.array([m.predict(X, verbose=0).reshape(-1) for m in models])
    predictions = predictions.mean(axis=0) > 0.500
    
    accuracy = np.mean(predictions == Y)
    return accuracy


###########################################################################################
# TEST MODELS
###########################################################################################

print("SHAPES", X_train.shape, Y_train.shape)

results = np.zeros((len(hidden_layer_options) * len(regularization_options), 3), 
                   dtype=object)

count = 0
for hidden_layer_sizes in hidden_layer_options:
    
    for regularization in regularization_options:
        
        accuracy = eval_models(hidden_layer_sizes, regularization)
        
        results[count] = [str(hidden_layer_sizes), str(regularization), accuracy]

        count += 1

        # Save the array
        np.save("../results/model_type_results_1.npy", results)
