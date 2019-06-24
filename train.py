
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.utils.np_utils import to_categorical
import keras.layers as kl
from keras.models import Model
from keras import regularizers

from keras.layers import LSTM, Input, Dense
from keras.models import Model
#from keras.layers import LeakyReLU

from keras.callbacks import EarlyStopping

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def compile_model(network):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    hidden = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    l2_rr = 0.003
    l2_r = 0.003
    d = 0.2
    r_d = 0.1

    lstm_act = 'relu'
    act = 'relu'

    input_data = kl.Input(shape= (4,10))
    lstm = kl.LSTM(hidden, input_shape=(4,10), return_sequences=True, activity_regularizer=regularizers.l2(l2_r),\
                   recurrent_regularizer=regularizers.l2(l2_rr), dropout=d, recurrent_dropout=r_d ,activation=activation)(input_data)
    #perc = kl.Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(l2_r))(lstm)
    lstm2 = kl.LSTM(hidden, activity_regularizer=regularizers.l2(l2_r), recurrent_regularizer=regularizers.l2(l2_rr),\
                    dropout=d, recurrent_dropout=r_d , activation=activation , return_sequences=True)(lstm)
    lstm3 = kl.LSTM(hidden, activity_regularizer=regularizers.l2(l2_r), recurrent_regularizer=regularizers.l2(l2_rr),\
                    dropout=d, recurrent_dropout=r_d , activation=activation , return_sequences=True)(lstm2)
    lstm4 = kl.LSTM(hidden, activity_regularizer=regularizers.l2(l2_r), recurrent_regularizer=regularizers.l2(l2_rr),\
                    dropout=d, recurrent_dropout=r_d , activation=activation ,return_sequences=True)(lstm3)
    lstm5 = kl.LSTM(hidden, activity_regularizer=regularizers.l2(l2_r), recurrent_regularizer=regularizers.l2(l2_rr),\
                    dropout=d, recurrent_dropout=r_d , activation=activation ,return_sequences=False)(lstm4)
    out = kl.Dense(1, activation= activation, activity_regularizer=regularizers.l2(l2_r))(lstm5)

    model = Model(input_data, out)
    #model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    model.compile(optimizer= optimizer, loss="mean_squared_error", metrics=["mse"])
    return model

def train_and_score(network, x_train , y_train , x_val,y_val, x_test , y_test , batch_size):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """

    model = compile_model(network)
    #              validation_data=(x_val, y_val),

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              callbacks=[early_stopper])

    # Calculate the RMSE score as fitness score for GA
    y_pred = model.predict(x_test , batch_size)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse  # 1 is accuracy. 0 is loss.
