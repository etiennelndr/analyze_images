try:
    from os import makedirs
    from os.path import exists

    # Importing required Keras modules containing model and layers
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Concatenate, Activation, concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.optimizers import Adam, RMSprop, SGD

    from tensorflow.nn import relu

    import numpy as np

    from keras.models import model_from_json
except ImportError as err:
    exit(err)

class NNModel(object):
    """
    Neural network model.
    """

    # Different model types
    MODEL_TYPES = {
        'sequential': Sequential, 
        'model'     : Model
    }

    # Constructor
    def __init__(self, model_type="sequential", data_to_process="digits"):
        """
        Initialization of the model.
        """
        self._model_type      = model_type
        self._data_to_process = data_to_process
        # History of model training
        self._history = None
        # Training state is set to False
        self._training = False
        # Initialize the main model
        self._initModel()

    def _initModel(self):
        """
        Initializes the main model.
        """
        if self._model_type not in self.MODEL_TYPES:
            raise NotImplementedError('Unknown model type: {}'.format(self._model_type))

        self._model = self.MODEL_TYPES[self._model_type]

    def addLayer(self, layer):
        """
        Adds a new layer to a sequential model.
        """
        self._model.add(layer)

    def rollback(self):
        """
        Re-initializes the model (~ rollback).
        """
        self._initModel()

    # Getters
    def getModelType(self):
        """
        Returns model name.
        """
        return self._model_type

    def getModel(self):
        """
        Returns an instance of the model.
        """
        return self._model

    def setModel(self, model):
        """
        Sets a new value to the current model.
        """
        self._model = model

    def getHistory(self):
        """
        Returns the history of model training.
        """
        return self._history

    def isTraining(self):
        """
        Returns the training state of the model. It returns True if the model
        is currently training, otherwise False.
        """
        return self._training

    # Setters
    def setDataToProcess(self, data_to_process):
        """
        Sets a new value to the type of data to process.
        """
        self._data_to_process = data_to_process

    # Abstract methods
    def createLayers(self):
        """
        Creates each layer of the model.
        """
        raise NotImplementedError("Please implement this method.")

    def learn(self):
        """
        Compiles and fits a model, evaluation is optional.
        """
        raise NotImplementedError("Please implement this method.")

    def loadDataToPredict(self, filename):
        """
        Loads data to predict.
        """
        raise NotImplementedError("Please implement this method.")

    def predictValue(self):
        """
        Predicts a value with a given data.
        """
        raise NotImplementedError("Please implement this method.")

    def saveModel(self, basename="basename", dir="models"):
        """
        Saves a model.
        """
        architectureFilePath = basename + '.json'
        print('\t - Architecture of the neural network: ' + architectureFilePath)

        if not exists(dir):
            makedirs(dir) # Create a new directory if it doesn't exist

        # Add a slash at the end of the directory name
        dir += "/"

        with open(dir + architectureFilePath, 'wt') as json_file:
            architecture = self._model.to_json()
            json_file.write(architecture)

        weightsFilePath = dir + basename + '.hdf5'
        print('\t - Weights of synaptic connections: ' + weightsFilePath)
        self._model.save(weightsFilePath)

    def openModel(self, architectureFileName, weightsFileName):
        """
        Opens an existing model.
        """
        if not exists(architectureFileName):
            print("ERROR: " + architectureFileName + " doesn't exist.")
            return
        elif architectureFileName[-4:] != "json":
            print("ERROR: architecture file extension MUST BE json.")
            return
        
        if not exists(weightsFileName):
            print("ERROR: " + weightsFileName + " doesn't exist.")
            return
        elif weightsFileName[-4:] != "hdf5":
            print("ERROR: weights file extension MUST BE hdf5.")
            return

        json_file = open(architectureFileName)
        architecture = json_file.read()
        json_file.close()
        # Create a model from a json file
        self._model = model_from_json(architecture)
        # Load weights
        self._model.load_weights(weightsFileName)

    def conv2d(self, inputs, filters, kernel_size=(3,3), action=None, pool_size=(2,2), up_size=(2,2)):
        """
        Creates and returns a layer with multiple convolution, dropout, up-sampling, max-pooling, etc layers.

        TODO: finish this method.
        """
        if action=="upSampling":
            # MxN Up Sampling
            up     = UpSampling2D(up_size)(inputs)
            # Concatenation
            concat = Concatenate(axis=3)([up, concatLayer])
            # MxN Convolution
            conv   = Conv2D(filters, kernel_size, padding='same', data_format='channels_last')(concat)
        else:
            # MxN Convolution
            conv   = Conv2D(filters, kernel_size, padding='same', data_format='channels_last')(inputs)

        print("conv:", conv.shape)
        bn     = BatchNormalization()(conv)
        act    = Activation(relu)(bn)
        # Dropout of 0.2
        drop   = Dropout(0.2)(act)
        # MxN Convolution
        conv   = Conv2D(filters, kernel_size, padding='same', data_format='channels_last')(drop)
        print("conv:", conv.shape)
        bn     = BatchNormalization()(conv)
        act    = Activation(relu)(bn)

        if action=="maxPooling":
            # MxN Max Pooling
            pool   = MaxPooling2D(pool_size=pool_size)(act)

    @staticmethod
    def _dice_coef(y_true, y_pred, smooth=1):
        """
        From: https://github.com/keras-team/keras/issues/3611 and https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
        """
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    @staticmethod
    def _dice_coef_loss(y_true, y_pred):
        """
        From: https://github.com/keras-team/keras/issues/3611 and https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
        """
        return 1 - dice_coef(y_true, y_pred)

    @staticmethod
    def _soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
        """
        From: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08

        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.
  
        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
            epsilon: Used for numerical stability to avoid divide by zero errors
    
        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation 
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        """
    
        # Skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * np.sum(y_pred * y_true, axes)
        denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
        return 1 - np.mean(numerator / (denominator + epsilon)) # Average over classes and batch

if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")
