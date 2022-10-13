from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Lambda, Conv1D
from tensorflow.keras import Model

def make_dense_model(input_size, hidden_layer_sizes, regularization=None, activation="relu"):
        
    input_layer = Input(shape=(input_size))
    
    hidden_layer = input_layer # To start the loop
    for hidden_size in hidden_layer_sizes:
        hidden_layer = Dense(hidden_size, activation=activation, kernel_regularizer=regularization)(hidden_layer)
    
    output_layer = Dense(1, activation="sigmoid")(hidden_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
    
    