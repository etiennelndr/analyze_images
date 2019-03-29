try:
    from .model import NNModel

    from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.optimizers import Adam
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.datasets import mnist

    import tensorflow as tf
    import numpy as np

    from PIL import Image
except ImportError as err:
    exit(err)

class DigitsModel(NNModel):
    """
    Neural network model for digit classification.
    """

    def __init__(self):
        """
        Initialization of the model.
        """
        NNModel.__init__(self, 'sequential', 'digits')

        # Input data shape
        self.input_shape = (28, 28, 1)

    def createLayers(self):
        """
        Creates each layer of the model.
        """
        # First of all, retrieve datas from MNIST dataset
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        # Reshaping the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape((x_train.shape[0],) + self.input_shape)
        x_val   = x_val.reshape((x_val.shape[0],) + self.input_shape)
        # Making sure that the values are float so that we can get decimal point
        # after division
        x_train = x_train.astype('float32')
        x_val   = x_val.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value
        x_train /= 255
        x_val   /= 255

        # Add layers to the model
        self.addLayer(Conv2D(36, kernel_size=(3,3), input_shape=self.input_shape))
        self.addLayer(MaxPooling2D(pool_size=(2,2)))
        self.addLayer(Conv2D(28, kernel_size=(2,2)))
        self.addLayer(MaxPooling2D(pool_size=(2,1)))
        self.addLayer(Flatten())
        self.addLayer(Dense(128, activation=tf.nn.relu))
        self.addLayer(Dropout(0.2))
        self.addLayer(Dense(10, activation=tf.nn.softmax))

        self.datas = { "x_train" : x_train, "y_train" : y_train, "x_val" : x_val, "y_val" : y_val }

        # Print the model summary
        self.getModel().summary()

    def learn(self):
        """
        Compiles and fits a model, evaluation is optional.
        """
        # Starting the training
        self._training = True

        # Number of epochs
        epochs = 10
        # Learning rate
        learning_rate = 1e-3
        # Compiling the model with an optimizer and a loss function
        self._model.compile(optimizer=Adam(lr=learning_rate, decay=learning_rate/epochs),
                        loss=[sparse_categorical_crossentropy],
                        metrics=[sparse_categorical_accuracy])

        if "x_val" in self.datas and "y_val" in self.datas:
            # Fitting the model by using our train and validation data
            # It returns the history that can be plot in the future
            self._history = self._model.fit(x=self.datas["x_train"], y=self.datas["y_train"], 
                                            epochs=epochs, 
                                            validation_data=(self.datas["x_val"], self.datas["y_val"]))
        else:
            # Fitting the model by using our train data
            # It returns the history that can be plot in the future
            self._history = self._model.fit(x=self.datas["x_train"], y=self.datas["y_train"], epochs=epochs)

        if "x_test" in self.datas and "y_test" in self.datas:
            # Evaluate the model
            self._model.evaluate(self.datas["x_test"], self.datas["y_test"])

        # Training is over
        self._training = False

    def loadDataToPredict(self, filename):
        """
        Loads data to predict.
        """
        # Open the desired picture
        im = Image.open(filename)
            
        # Convert to a black and white picture
        im = im.convert("L")
        # Resize the picture
        im = im.resize(self.input_shape[:2])

        # Get the image array
        self.__imgToPredict = np.array(im)

        # Binarize this array
        self.__imgToPredict[self.__imgToPredict >  40] = 15
        self.__imgToPredict[self.__imgToPredict <= 40] = 240

        # Be careful -> each pixel value must be a float
        self.__Data = self.__imgToPredict.astype('float32')
        # Make a copy of this array to show the picture
        img = np.copy(self.__imgToPredict)

        # Normalize the image
        self.__imgToPredict /= 255

        # Close the file pointer (if possible)
        im.close()

    def predictValue(self):
        """
        Predicts a value with a given data.
        """
        return str(self._model.predict(self.__imgToPredict.reshape((1,) + self.input_shape)).argmax())
