from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def model(image_size):

    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=image_size))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model
