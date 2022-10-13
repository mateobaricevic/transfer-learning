from keras.layers import (LSTM, Activation, Bidirectional, Conv1D, Dense,
                          Dropout, Flatten, Input, MaxPooling1D)
from keras.models import Model


##############
#   MagNet   #
##############
def magnet(waveform_length, num_classes):
    waveform_input = Input(shape=(waveform_length, 1), name='waveform_input')
    
    e = Conv1D(64, 3, padding = 'same', name='conv1d')(waveform_input)
    e = Dropout(0.2)(e, training=True)
    e = MaxPooling1D(4, padding='same')(e)
     
    e = Conv1D(64, 3, padding = 'same', name='conv1d_1')(e)
    e = Dropout(0.2)(e, training=True)
    e = MaxPooling1D(4, padding='same')(e)
     
    e = Conv1D(64, 3, padding = 'same', name='conv1d_2')(e)
    e = Dropout(0.2)(e, training=True)
    e = MaxPooling1D(4, padding='same')(e)
     
    e = Conv1D(64, 3, padding = 'same', name='conv1d_3')(e)
    e = Dropout(0.2)(e, training=True)
    e = MaxPooling1D(4, padding='same')(e)

    e = Bidirectional(LSTM(32, return_sequences=True))(e)
    e = Flatten()(e)
    e = Dense(num_classes)(e)

    o = Activation('softmax', name='output_layer')(e)
    
    return Model(inputs=[waveform_input], outputs=o)
