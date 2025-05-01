'''
The following code is designed for the MNIST handwritten digit classification, please
refer to Lecture 8
a. Complete the “evaluate_model” function with one statement of invoking
“define_model”
b. Write a small code to call the functions of “load_dataset”, “prep_pixels” and
“evaluate_model”
c. Save the classification model into your local drive
d. Download “sample_image” file from the Week 8 folder in BBL into your local drive
e. Below is a code fragment for loading the image of “sample_image” from your local
drive, write a small code to load the saved classification model in (c) above and
predict “sample_image”

'''

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import numpy as np

# load and prepare the image
def load_image(sample_image):
    # load the image
   img = load_img(sample_image, color_mode='grayscale', target_size=(28, 28))
   # convert to array
   img = img_to_array(img)
   # reshape into a single sample with 1 channel
   img = img.reshape(1, 28, 28, 1)
   # prepare pixel data
   img = img.astype('float32')
   img = img / 255.0
   return img

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# define cnn model
def define_model():
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
    #                  input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(10, activation='softmax'))
    # # compile model
    # opt = SGD(lr=0.01, momentum=0.9)
    # model.compile(optimizer=opt, loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # return model

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))  # Explicit Input layer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        #model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix],dataX[test_ix], dataY[test_ix]

        # fit model
        model = define_model()
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)

        # Save final trained model
        model.save('final_model.h5')
    return scores, histories

# load an image and predict the class
def run_example():
    # load the image
    img = load_image('sample_image.png')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    prediction = model.predict(img)
    digit = np.argmax(prediction)  # pick class with highest probability
    print(f"Predicted Digit: {digit}")


# Main code to execute all
if __name__ == "__main__": # predefined variable in python which works like entry point.
    print(__name__)
    print('__main__')
# a, b) Train and evaluate model
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)


    # entry point, run the example
    run_example()
