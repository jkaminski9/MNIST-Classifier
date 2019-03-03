from keras.datasets import mnist
from model import model
from keras.utils import to_categorical

#Load / Process the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshape for the CNN
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#generate / compile the model
image_size = x_train[0].shape
model = model(image_size)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, epochs = 3, validation_split=0.2, batch_size=32)

#test the model
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print 'Testing Results: ', loss_and_metrics
