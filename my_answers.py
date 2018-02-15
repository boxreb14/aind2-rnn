import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i:i+window_size] for i in range(len(series)-window_size)]
    y = [series[i+window_size] for i in range(len(series)-window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model=Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = [' ', '!', ',', '.', ':', ';', '?']
    chars = sorted(list(set(text)))
    valid_chars = list(string.ascii_lowercase)
    valid_chars += punctuation
    invalid_chars = set(chars) - set(valid_chars)
    for v in invalid_chars:
        text = text.replace(v, ' ')
    #text = text.replace('"',' ')
    #text = text.replace('$',' ')
    #text = text.replace('%',' ')
    #text = text.replace('&',' ')
    #text = text.replace("'",' ')
    #text = text.replace('(',' ')
    #text = text.replace(')',' ')
    #text = text.replace('*',' ')
    #text = text.replace('-',' ')
    #text = text.replace('/',' ')
    #text = text.replace('@',' ')
    #text = text.replace('~',' ')
    #text = text.replace('^',' ')
    #text = text.replace('<',' ')
    #text = text.replace('>',' ')
    #text = text.replace('+',' ')
    #text = text.replace('{',' ')
    #text = text.replace('}',' ')
    #text = text.replace('[',' ')
    #text = text.replace(']',' ')
    #text = text.replace('|',' ')
    #text = text.replace('_',' ')
    #text = text.replace('=',' ')
    #text = text.replace('`',' ')
    #text = text.replace('0',' ')
    #text = text.replace('1',' ')
    #text = text.replace('2',' ')
    #text = text.replace('3',' ')
    #text = text.replace('4',' ')
    #text = text.replace('5',' ')
    #text = text.replace('6',' ')
    #text = text.replace('7',' ')
    #text = text.replace('8',' ')
    #text = text.replace('9',' ')
    #text = text.replace('à',' ')
    #text = text.replace('â',' ')
    #text = text.replace('è',' ')
    #text = text.replace('é',' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:i+window_size] for i in range(0, len(text)-window_size, step_size)]
    outputs = [text[i+window_size] for i in range(0, len(text)-window_size, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model=Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    return model
