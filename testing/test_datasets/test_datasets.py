#!/usr/bin/env python3
#########################################################################################################
# IMPORTS
#########################################################################################################

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Lambda, Conv1D
from tensorflow.keras import Model

from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("../")
from tensorflow_gpu.configure import configure_tensorflow
configure_tensorflow(eager=False, gpu_index=1, allow_growth_gpu=False)


#########################################################################################################
# MAKE MODEL FUNCTIONS
#########################################################################################################

def make_dense_model(input_size, hidden_layer_sizes, regularization=None, activation="relu"):
        
    input_layer = Input(shape=(input_size))
    
    hidden_layer = input_layer # To start the loop
    for hidden_size in hidden_layer_sizes:
        hidden_layer = Dense(hidden_size, activation=activation, 
                             kernel_regularizer=regularization)(hidden_layer)
    
    output_layer = Dense(1, activation="sigmoid")(hidden_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


#########################################################################################################
# DATA LOADING FUNCTIONS
#########################################################################################################

# Define a function to load the data in
data_path = "../data/"

def load_single_mean(lookback_game):
    X = np.load(data_path + f"single_mean_lg{lookback_game}_X.npy")
    Y = np.load(data_path + f"single_mean_lg{lookback_game}_Y.npy")
    ids = np.load(data_path + f"single_mean_lg{lookback_game}_ids.npy")
    return X, Y, ids

def load_single_diff(lookback_game):
    X = np.load(data_path + f"single_diff_lg{lookback_game}_X.npy")
    Y = np.load(data_path + f"single_diff_lg{lookback_game}_Y.npy")
    ids = np.load(data_path + f"single_diff_lg{lookback_game}_ids.npy")
    return X, Y, ids

def load_double_mean(lookback_game):
    X = np.load(data_path + f"double_mean_lg{lookback_game}_X.npy")
    Y = np.load(data_path + f"double_mean_lg{lookback_game}_Y.npy")
    ids = np.load(data_path + f"double_mean_lg{lookback_game}_ids.npy")
    return X, Y, ids

def load_double_diff(lookback_game):
    X = np.load(data_path + f"single_diff_lg{lookback_game}_X.npy")
    Y = np.load(data_path + f"single_diff_lg{lookback_game}_Y.npy")
    ids = np.load(data_path + f"single_diff_lg{lookback_game}_ids.npy")
    return X, Y, ids


#########################################################################################################
# MODEL EVALUATION FUNCTIONS
#########################################################################################################

def eval_models(X, Y, ids, n_models, is_single, model_name):
    
    train_pct, test_pct, val_pct = 0.6, 0.2, 0.2
    
    # Split the data
    X_train = X[:int(train_pct*X.shape[0])]
    X_test = X[int(train_pct*X.shape[0]):int((train_pct+test_pct)*X.shape[0])]
    X_val = X[-int(val_pct*X.shape[0]):]
    
    Y_train = Y[:int(train_pct*X.shape[0])]
    Y_test = Y[int(train_pct*X.shape[0]):int((train_pct+test_pct)*X.shape[0])]
    Y_val = Y[-int(val_pct*X.shape[0]):]
    
    ids_train = ids[:int(train_pct*X.shape[0])]
    ids_test = ids[int(train_pct*X.shape[0]):int((train_pct+test_pct)*X.shape[0])]
    ids_val = ids[-int(val_pct*X.shape[0]):]
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    # Loop through and train models
    trained_models = []
    for _ in range(n_models):
        
        # Define the model structure
        model = make_dense_model(input_size=X.shape[1], hidden_layer_sizes=[10], 
                                 regularization=None, activation="relu")
        
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
        
        # Print out a quick evaluation
        val_acc = model.evaluate(X_val, Y_val, verbose=0)
        val_acc = {out: val_acc[i] for i, out in enumerate(trained_models[0].metrics_names)}
        print(f"Validation Accuracy: {val_acc['accuracy']*100:.2f}%")
        
    # Evaluate all the models together
    if is_single:
        final_accuracy = evaluate_single_models(trained_models, X_val, Y_val, ids_val)
    else:
        final_accuracy = evaluate_double_models(trained_models, X_val, Y_val)
        
    print(f"{model_name} Final Accuracy: {final_accuracy*100:.2f}%.")
    return final_accuracy
        

    
def evaluate_single_models(models, X, Y, ids):
    
    # Generate predictions
    predictions = np.array([m.predict(X, verbose=0).reshape(-1) for m in models])
    predictions = predictions.mean(axis=0) > 0.500
    
    # Use ids to find the same games
    right, wrong = 0, 0
    for game_id in np.unique(ids):
        game_mask = ids == game_id
        
        if np.sum(game_mask) != 2:
            continue
        
        game_preds = predictions[game_mask]
        game_Y = Y[game_mask]
        
        if game_Y[np.argmax(game_preds)]:
            right += 1
        else:
            wrong += 1
            
    accuracy = right / (right + wrong)
    return accuracy
    
def evaluate_double_models(models, X, Y):
    
    # Generate predictions
    predictions = np.array([m.predict(X, verbose=0).reshape(-1) for m in models])
    predictions = predictions.mean(axis=0) > 0.500
    
    accuracy = np.mean(predictions == Y)
    return accuracy
        
    
#########################################################################################################
# TESTING PARAMTERS
#########################################################################################################

lookback_games = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
n_models = 21

learning_rate = 1e-4
epochs = 300 # Should always stop early.


#########################################################################################################
# CALCULATE ACCURACIES
#########################################################################################################

all_accuracies = np.zeros((len(lookback_games), 4+1))
for i, lookback_game in enumerate(lookback_games):
    
#     # Single Mean Data
#     X, Y, ids = load_single_mean(lookback_game)
#     single_mean_accuracy = eval_models(X, Y, ids, n_models, is_single=True, model_name="Single Mean")
    
#     # Single Diff Data
#     X, Y, ids = load_single_diff(lookback_game)
#     single_diff_accuracy = eval_models(X, Y, ids, n_models, is_single=True, model_name="Single Diff")
    
    # Double Mean Data
    X, Y, ids = load_double_mean(lookback_game)
    double_mean_accuracy = eval_models(X, Y, ids, n_models, is_single=False, model_name="Double Mean")
    
#     # Double Diff Data
#     X, Y, ids = load_double_diff(lookback_game)
#     double_diff_accuracy = eval_models(X, Y, ids, n_models, is_single=False, model_name="Double Diff")
    
    # Track them (first column stores the lookback game value)
#     all_accuracies[i] = [lookback_game, single_mean_accuracy, single_diff_accuracy, double_mean_accuracy, 
#                          double_diff_accuracy]
    all_accuracies[i] = [lookback_game, 0, 0, double_mean_accuracy, 0] # Know what one the best is
    print(f"Done training lookback_game={lookback_game}.")


    #####################################################################################################
    # SAVE THE RESULTS
    #####################################################################################################
    np.save("../results/data_type_results_11_16.npy", all_accuracies)

