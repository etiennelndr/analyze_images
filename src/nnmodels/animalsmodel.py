try:
    from model import NNModel

    from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
    from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
    from keras.optimizers import Adam
    from keras.losses import categorical_crossentropy
    from keras.metrics import categorical_accuracy

    import tensorflow as tf
    import numpy as np

    from PIL import Image
except ImportError as err:
    exit(err)

class AnimalsModel(NNModel):
    """
    Neural network model for animal classification.
    """

    def __init__(self):
        """
        Initialization of the model.
        """
        NNModel.__init__(self, 'sequential', 'animals')

        # Input data shape
        self.input_shape = (150, 150, 3)

    def createLayers(self):
        """
        Creates each layer of the model.
        """
        #base_dir  = "C:/Users/e_sgouge/Documents/Etienne/Python/analyze_images/datas/dogs_vs_cats"
        base_dir  = "D:/Documents/Programmation/Python/analyze_images/datas/dogs_vs_cats"
        train_dir = join(base_dir, "training")
        val_dir   = join(base_dir, "validation")
        test_dir  = join(base_dir, "testing")

        train_datagen = ImageDataGenerator(rescale=1./255,
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')
        val_datagen   = ImageDataGenerator(rescale=1./255)
        test_datagen  = ImageDataGenerator(rescale=1./255)

        # Generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='binary')
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='binary')
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=20,
            class_mode='binary')
            
        # Datas
        self.datas = { "train_generator" : train_generator, "val_generator" : val_generator, "test_generator" : test_generator }

        # Add layers to the model
        self.addLayer(Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=self.input_shape))
        self.addLayer(MaxPooling2D((2, 2)))
        self.addLayer(Conv2D(64, (3, 3), activation=tf.nn.relu))
        self.addLayer(MaxPooling2D((2, 2)))
        self.addLayer(Conv2D(128, (3, 3), activation=tf.nn.relu))
        self.addLayer(MaxPooling2D((2, 2)))
        self.addLayer(Conv2D(128, (3, 3), activation=tf.nn.relu))
        self.addLayer(MaxPooling2D((2, 2)))
        self.addLayer(Flatten())
        self.addLayer(Dropout(0.5))
        self.addLayer(Dense(512, activation=tf.nn.relu))
        self.addLayer(Dense(1, activation=tf.nn.sigmoid))

        self.getModel().summary()

    def learn(self, data, epochs=100):
        """
        Compiles and fits a model, evaluation is optional.
        """
        # Compiling the model with an optimizer and a loss function
        self._model.compile(optimizer=Adam(lr=1e-3),
                        loss=categorical_crossentropy,#[sparse_categorical_crossentropy],
                        metrics=[categorical_accuracy])#[sparse_categorical_accuracy])

        # Fitting the model by using our train and validation data
        # It returns the history that can be plot in the future
        if "train_generator" in data and "val_generator" in data:
            # Fit including validation datas
            self._history = self._model.fit_generator(
                data["train_generator"],
                steps_per_epoch = 100,
                epochs = epochs,
                validation_data = data["val_generator"],
                validation_steps = 20)
        elif "train_generator" in data:
            # Fit without validation datas
            self._history = self._model.fit_generator(
                data["train_generator"],
                steps_per_epoch = 100,
                epochs = epochs,
                validation_steps = 20)
        else:
            raise NotImplementedError("Unknown data")

        if "test_generator" in data:
            # Evaluation of the model
            self._model.evaluate_generator(data["test_generator"], steps=50, verbose=1)

    def loadDataToPredict(self, filename):
        """
        Loads data to predict.
        """
        # Open the desired picture
        im = Image.open(filename)
        
        # Resize the picture
        im = im.resize(self.input_shape[:2])

        # Get the image array
        self.__imgToPredict = np.array(im)

        # Be careful -> each pixel value must be a float
        self.__imgToPredict = self.__imgToPredict.astype('float32')
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
        pred = self._model.predict(self.__imgToPredict.reshape((1,) + self.input_shape))
        return "dog" if pred[0][0] >= 0.5 else "cat"
