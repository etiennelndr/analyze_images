try:
    from .model import NNModel

    from PIL import Image
    from scipy.misc import imsave, imresize

    # Importing the required Keras modules containing models, layers, optimizers, losses, etc
    from keras.models import Model
    from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Concatenate, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Reshape, Permute
    from keras.preprocessing.image import img_to_array, load_img, array_to_img
    from keras.optimizers import Adam, RMSprop
    from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
    from keras.metrics import categorical_accuracy, sparse_categorical_accuracy, binary_accuracy

    from os import listdir
    from os.path import isfile, exists, join, realpath
    from os.path import split as pathsplit
    from random import randint

    import tensorflow as tf

    import matplotlib.pyplot as plt

    import numpy as np

    from image import transfromXY
except ImportError as err:
    exit(err)

class AerialRoadsModel(NNModel):
    """
    Neural network model for roads segmentation in aerial images.

    Sources:
        - roads dataset     -> https://www.cs.toronto.edu/~vmnih/data/          # Not implemented
    """

    def __init__(self):
        """
        Initialization of the model.
        """
        NNModel.__init__(self, "model", "aerial_roads")

        # Number of classes to segment
        # 0 -> not a building
        # 1 -> a building
        self.__nClasses  = 1
        # Input data shape
        self.input_shape = (336, 336, 3)
        # File extensions for data to predict
        self.FILE_EXTENSIONS = [
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ]

    def createLayers(self):
        """
        Creates each layer of the model.
        """
        base_dir  = join(realpath(__file__).split("src")[0], "datas/aerial_roads")
        train_dir = join(base_dir, "training")
        val_dir   = join(base_dir, "validation")
        test_dir  = join(base_dir, "testing")

        assert exists(train_dir) == True
        assert exists(val_dir)   == True
        assert exists(test_dir)  == True

        def createGenerator(dir, batch_size=2):
            x_dir = join(dir, "x")
            y_dir = join(dir, "y")

            assert exists(x_dir) == True
            assert exists(y_dir) == True

            x_files = [join(x_dir, n) for n in listdir(x_dir) if isfile(join(x_dir, n))]
            y_files = [join(y_dir, n) for n in listdir(y_dir) if isfile(join(y_dir, n))]
            
            assert len(x_files) == len(y_files)
            
            while True:
                x, y = list(), list()
                for _ in range(batch_size):
                    # Get a random index between 0 and len(x_files)
                    index = randint(0, len(x_files)-1)

                    # MUST be true (files must have the same name)
                    assert pathsplit(x_files[index].split(".")[0])[-1] == pathsplit(y_files[index].split(".")[0])[-1]

                    x_img = img_to_array(load_img(x_files[index]))
                    y_img = img_to_array(load_img(y_files[index]))

                    # Resize each image
                    x_img, y_img = imresize(x_img, self.input_shape[:2]), imresize(y_img, self.input_shape[:2])
                    # Apply a transformation on these images
                    x_img, y_img = transfromXY(x_img, y_img)

                    # Change y shape : (m, n, 3) -> (m, n, 2) (2 is the class number)
                    temp_y_img = np.zeros(self.input_shape[:2] + (1,))
                    temp_y_img[y_img[:,:,0] == 0]   = 0
                    temp_y_img[y_img[:,:,0] == 255] = 1
                    y_img = temp_y_img

                    # Convert to float
                    x_img = x_img.astype('float32')
                    y_img = y_img.astype('float32')
                    # Divide by the maximum value of each pixel
                    x_img /= 255
                    # Append images to the lists
                    x.append(x_img)
                    y.append(y_img)
                yield np.array(x), np.array(y)

        # Create a generator for each step
        train_generator = createGenerator(train_dir, 4)
        val_generator   = createGenerator(val_dir,   4)
        test_generator  = createGenerator(test_dir,  4)

        # Datas
        self.datas = { "train_generator": train_generator, "val_generator": val_generator, "test_generator": test_generator }

        # Inputs
        inputs  = Input(self.input_shape)
        # ----- First Convolution - Max Pooling -----
        # 3x3 Convolution
        conv1  = Conv2D(16, (3, 3), padding='same', data_format='channels_last', name='conv1_1')(inputs)
        print("conv1:", conv1.shape)
        bnor1  = BatchNormalization(name='bnor1_1')(conv1)
        acti1  = Activation(tf.nn.relu, name='acti1_1')(bnor1)
        # Dropout of 0.2
        drop1  = Dropout(0.2, name='drop1_1')(acti1)
        # 3x3 Convolution
        conv1  = Conv2D(16, (3, 3), padding='same', data_format='channels_last', name='conv1_2')(drop1)
        print("conv1:", conv1.shape)
        bnor1  = BatchNormalization(name='bnor1_2')(conv1)
        acti1  = Activation(tf.nn.relu, name='acti1_2')(bnor1)
        # 2x2 Max Pooling
        pool1  = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(acti1)

        # ----- Second Convolution - Max Pooling -----
        # 3x3 Convolution
        conv2  = Conv2D(32, (3, 3), padding='same', data_format='channels_last', name='conv2_1')(pool1)
        print("conv2:", conv2.shape)
        bnor2  = BatchNormalization(name='bnor2_1')(conv2)
        acti2  = Activation(tf.nn.relu, name='acti2_1')(bnor2)
        # Dropout of 0.2
        drop2  = Dropout(0.2, name='drop2_1')(acti2)
        # 3x3 Convolution
        conv2  = Conv2D(32, (3, 3), padding='same', data_format='channels_last', name='conv2_2')(drop2)
        print("conv2:", conv2.shape)
        bnor2  = BatchNormalization(name='bnor2_2')(conv2)
        acti2  = Activation(tf.nn.relu, name='acti2_2')(bnor2)
        # 2x2 Max Pooling
        pool2  = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(acti2)

        # ----- Third Convolution - Max Pooling -----
        # 3x3 Convolution
        conv3  = Conv2D(64, (3, 3), padding='same', data_format='channels_last', name='conv3_1')(pool2)
        print("conv3:", conv3.shape)
        bnor3  = BatchNormalization(name='bnor3_1')(conv3)
        acti3  = Activation(tf.nn.relu, name='acti3_1')(bnor3)
        # Dropout of 0.2
        drop3  = Dropout(0.2, name='drop3_2')(acti3)
        # 3x3 Convolution
        conv3  = Conv2D(64, (3, 3), padding='same', data_format='channels_last', name='conv3_2')(drop3)
        print("conv3:", conv3.shape)
        bnor3  = BatchNormalization(name='bnor3_2')(conv3)
        acti3  = Activation(tf.nn.relu, name='acti3_2')(bnor3)
        # 2x2 Max Pooling
        pool3  = MaxPooling2D(pool_size=(2, 2), name='pool3_1')(acti3)

        # ----- Fourth Convolution - Max Pooling -----
        # 3x3 Convolution
        conv4  = Conv2D(128, (3, 3), padding='same', data_format='channels_last', name='test_conv4_1')(pool3)
        print("conv4:", conv4.shape)
        bnor4  = BatchNormalization(name='test_bnor4_1')(conv4)
        acti4  = Activation(tf.nn.relu, name='test_acti4_1')(bnor4)
        # Dropout of 0.2
        drop4  = Dropout(0.2, name='test_drop4_2')(acti4)
        # 3x3 Convolution
        conv4  = Conv2D(128, (3, 3), padding='same', data_format='channels_last', name='test_conv4_2')(drop4)
        print("conv4:", conv4.shape)
        bnor4  = BatchNormalization(name='test_bnor4_2')(conv4)
        acti4  = Activation(tf.nn.relu, name='test_acti4_2')(bnor4)
        # 2x2 Max Pooling
        pool4  = MaxPooling2D(pool_size=(2, 2), name='test_pool4_1')(acti4)

        # ----- Fifth Convolution -----
        # 3x3 Convolution
        conv5  = Conv2D(256, (3, 3), padding='same', data_format='channels_last', name='conv5_1')(pool4)
        print("conv5:", conv5.shape)
        bnor5  = BatchNormalization(name='bnor5_1')(conv5)
        acti5  = Activation(tf.nn.relu, name='acti5_1')(bnor5)
        # Dropout of 0.25
        drop5  = Dropout(0.25, name='drop5_1')(acti5)
        # 3x3 Convolution
        conv5  = Conv2D(256, (3, 3), padding='same', data_format='channels_last', name='conv5_2')(drop5)
        print("conv5:", conv5.shape)
        bnor5  = BatchNormalization(name='bnor5_2')(conv5)
        acti5  = Activation(tf.nn.relu, name='acti5_2')(bnor5)

        # ----- Sixth Convolution -----
        # 2x2 Up Sampling
        upsp6  = UpSampling2D(size = (2,2), name='test_upsp6_1')(acti5)
        # Concatenation
        conc6  = Concatenate(axis=3, name='test_conc6_1')([upsp6, acti4])
        # 3x3 Convolution
        conv6  = Conv2D(128, (3, 3), padding='same', data_format='channels_last', name='test_conv6_1')(conc6)
        print("conv6:", conv6.shape)
        bnor6  = BatchNormalization(name='test_bnor6_1')(conv6)
        acti6  = Activation(tf.nn.relu, name='test_acti6_1')(bnor6)
        # Dropout of 0.2
        drop6  = Dropout(0.2, name='test_drop6_2')(acti6)
        # 3x3 Convolution
        conv6  = Conv2D(128, (3, 3), padding='same', data_format='channels_last', name='test_conv6_2')(drop6)
        print("conv6:", conv6.shape)
        bnor6  = BatchNormalization(name='test_bnor6_2')(conv6)
        acti6  = Activation(tf.nn.relu, name='test_acti6_2')(bnor6)

        # ----- Seventh Convolution - Up Sampling -----
        # 2x2 Up Sampling
        upsp7  = UpSampling2D(size = (2,2), name='upsp7_1')(acti6)
        # Concatenation
        conc7  = Concatenate(axis=3, name='conc7_1')([upsp7, acti3])
        # 3x3 Convolution
        conv7  = Conv2D(64, (3, 3), padding='same', data_format='channels_last', name='conv7_1')(conc7)
        print("conv7:", conv7.shape)
        bnor7  = BatchNormalization(name='bnor7_1')(conv7)
        acti7  = Activation(tf.nn.relu, name='acti7_1')(bnor7)
        # Dropout of 0.2
        drop7  = Dropout(0.2, name='drop7_2')(acti7)
        # 3x3 Convolution
        conv7  = Conv2D(64, (3, 3), padding='same', data_format='channels_last', name='conv7_2')(drop7)
        print("conv7:", conv7.shape)
        bnor7  = BatchNormalization(name='bnor7_2')(conv7)
        acti7  = Activation(tf.nn.relu, name='acti7_2')(bnor7)

        # ----- Eighth Convolution - Up Sampling -----
        # 2x2 Up Sampling
        upsp8  = UpSampling2D(size = (2,2), name='upsp8_1')(acti7)
        # Concatenation
        conc8  = Concatenate(axis=3, name='conc8_1')([upsp8, acti2])
        # 3x3 Convolution
        conv8  = Conv2D(32, (3, 3), padding='same', data_format='channels_last', name='conv8_1')(conc8)
        print("conv8:", conv8.shape)
        bnor8  = BatchNormalization(name='bnor8_1')(conv8)
        acti8  = Activation(tf.nn.relu, name='acti8_1')(bnor8)
        # Dropout of 0.2
        drop8  = Dropout(0.2, name='drop8_1')(acti8)
        # 3x3 Convolution
        conv8  = Conv2D(32, (3, 3), padding='same', data_format='channels_last', name='conv8_2')(drop8)
        print("conv8:", conv8.shape)
        bnor8  = BatchNormalization(name='bnor8_2')(conv8)
        acti8  = Activation(tf.nn.relu, name='acti8_2')(bnor8)

        # ----- Ninth Convolution - Up Sampling -----
        # 2x2 Up Sampling
        upsp9  = UpSampling2D(size = (2,2), name='upsp9_1')(acti8)
        # Concatenation
        conc9  = Concatenate(axis=3, name='conc9_1')([upsp9, acti1])
        # 3x3 Convolution
        conv9  = Conv2D(16, (3, 3), padding='same', data_format='channels_last', name='conv9_1')(conc9)
        print("conv9:", conv9.shape)
        bnor9  = BatchNormalization(name='bnor9_1')(conv9)
        acti9  = Activation(tf.nn.relu, name='acti9_1')(bnor9)
        # Dropout of 0.2
        drop9  = Dropout(0.2, name='drop9_1')(acti9)
        ## 3x3 Convolution
        conv9  = Conv2D(16, (3, 3), padding='same', data_format='channels_last', name='conv9_2')(drop9)
        print("conv9:", conv9.shape)
        bnor9  = BatchNormalization(name='bnor9_2')(conv9)
        acti9  = Activation(tf.nn.relu, name='acti9_2')(bnor9)

        # ----- Tenth Convolution (outputs) -----
        # 3x3 Convolution
        conv10 = Conv2D(2, (3, 3), padding='same', data_format='channels_last', name='conv10_1')(acti9)
        print("conv10:", conv10.shape)
        bnor10 = BatchNormalization(name='bnor10_1')(conv10)
        acti10 = Activation(tf.nn.sigmoid, name='acti10_1')(bnor10)
        # 1x1 Convolution
        conv10 = Conv2D(self.__nClasses, (1, 1), padding='same', data_format='channels_last', name='conv10_2')(acti10)
        print("conv10:", conv10.shape)
        bnor10 = BatchNormalization(name='bnor10_2')(conv10)
        acti10 = Activation(tf.nn.sigmoid, name='acti10_2')(bnor10)

        # Set a new model with the inputs and the outputs (tenth convolution)
        self.setModel(Model(inputs=inputs, outputs=acti10))

        # Get a summary of the previously create model
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
        learning_rate = 1e-4
        # Compiling the model with an optimizer and a loss function
        self._model.compile(optimizer=Adam(lr=learning_rate),#, decay=learning_rate/epochs),
                        loss=binary_crossentropy,
                        metrics=["accuracy"])

        # Fitting the model by using our train and validation data
        # It returns the history that can be plot in the future
        if "train_generator" in self.datas and "val_generator" in self.datas:
            # Fit including validation datas
            self._history = self._model.fit_generator(
                self.datas["train_generator"],
                steps_per_epoch = 1000,
                epochs = epochs,
                validation_data = self.datas["val_generator"],
                validation_steps = 100)
        elif "train_generator" in self.datas:
            # Fit without validation datas
            self._history = self._model.fit_generator(
                self.datas["train_generator"],
                steps_per_epoch = 1000,
                epochs = epochs)
        else:
            raise NotImplementedError("Unknown data")

        if "test_generator" in self.datas:
            # Evaluation of the model
            testLoss, acc_test = self._model.evaluate_generator(self.datas["test_generator"], steps=500, verbose=1)
            print("Loss / test: " + str(testLoss) + " and accuracy: " + str(acc_test))

        # Training is over
        self._training = False

    def loadDataToPredict(self, filename):
        """
        Loads data to predict.
        """
        # Store the file name
        self.__filename = filename

        # Open the desired picture
        im = Image.open(filename)

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
        # Get the array of the real image
        img_array = np.array(Image.open(self.__filename))
        # Store the real shape for later
        real_shape = img_array.shape

        # At this time we can only use images of shape (m*500, n*500, 3)
        assert real_shape[0]%500 == 0
        assert real_shape[1]%500 == 0

        # Predict the segmentation for this picture (its array is stored in data)
        pred = np.zeros(real_shape[:2] + (1,))
        for i in range(int(real_shape[0]/500)):
            for j in range(int(real_shape[1]/500)):
                print(i,j)
                # Get a sub-array of the main array
                sub_array = self.__imgToPredict[i*500:(i+1)*500:, j*500:(j+1)*500:, :]
                sub_img = array_to_img(sub_array).resize(self.input_shape[:2])
                # Because array_to_img is modifying array values to [0,255] we have 
                # to divide each value by 255
                sub_array = np.array(sub_img)/255.
                
                # Predict the segmentation for this sub-array
                pred_array = self._model.predict(sub_array.reshape((1,) + sub_array.shape))
                pred_img = array_to_img(pred_array.reshape(pred_array.shape[1:])).resize((500,500))
                pred_array = np.array(pred_img).reshape(500,500,1)
                # Add this sub-array to the main array
                pred[i*500:(i+1)*500:, j*500:(j+1)*500:, :] = pred_array/255.

        # Reshape the image array to (m, n, 3)
        reshaped_img_array = np.array(Image.fromarray(img_array).resize(real_shape[:2][::-1]))
        # If the result for the second value is more than 0.85 -> store a 
        # "green" array for this index
        reshaped_img_array[pred[:,:,0] > 0.85] = [0, 240, 0]
        # Because we need to put the segmented road on the real image, we have to
        # reshape the predicted array to the real shape
        reshaped_img_array = np.array(Image.fromarray(reshaped_img_array).resize(real_shape[:2][::-1]))
        # Now, for each element in the picture, replace it or not
        img_array[reshaped_img_array[:,:,1] == 240] = [0,240,0]

        # Create a new Image instance with the new_img_array array
        new_img = Image.fromarray(img_array.astype('uint8'))
        # Finally, save this image
        new_img.save("segmented_img.jpg")
        # Save the unsegmented image
        imsave("unsegmented_img.jpg", np.array(Image.open(self.__filename)))

        # Hold on, close the pointers before leaving
        new_img.close()

        print("Done")
