try:
    from os import makedirs
    from os.path import exists, join

    # Importing required Keras modules containing model and layers
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Concatenate, Activation, concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.nn import relu

    import keras.backend as K

    import numpy as np

    from keras.models import model_from_json
except ImportError as err:
    exit(err)


class NNModel(object):
    """
    Neural network model.
    """

    # Different model types
    _MODEL_TYPES = {
        'sequential': Sequential(), 
        'model'     : Model()
    }

    # Constructor
    def __init__(self, model_type="sequential", data_to_process="digits"):
        """
        Initialization of the model.
        """
        self._model_type      = model_type
        self._data_to_process = data_to_process
        # Model training history
        self._history         = None
        # Training state is set to False
        self._training        = False
        # File extensions for data to predict
        self.FILE_EXTENSIONS  = list()
        # Initialize the main model
        self.__init_model()

    def __init_model(self):
        """
        Initializes the main model.
        """
        if self._model_type not in self._MODEL_TYPES:
            raise NotImplementedError('Unknown model type: {}'.format(self._model_type))

        self._model = self._MODEL_TYPES[self._model_type]

    def add_layer(self, layer):
        """
        Adds a new layer to a sequential model.
        """
        self._model.add(layer)

    def rollback(self):
        """
        Re-initializes the model (~ rollback).
        """
        self.__init_model()

    # Getters
    def get_model_type(self):
        """
        Returns model name.
        """
        return self._model_type

    def get_model(self):
        """
        Returns an instance of the model.
        """
        return self._model

    def set_model(self, model):
        """
        Sets a new value to the current model.
        """
        self._model = model

    def get_history(self):
        """
        Returns the history of model training.
        """
        return self._history

    def is_training(self):
        """
        Returns the training state of the model. It returns True if the model
        is currently training, otherwise False.
        """
        return self._training

    # Setters
    def set_data_to_process(self, data_to_process):
        """
        Sets a new value to the type of data to process.
        """
        self._data_to_process = data_to_process

    def concatenate_extensions(self):
        """
        Concatenates all extensions in one sentence.
        """
        exts = ""
        for i in range(len(self.FILE_EXTENSIONS)):
            ext = self.FILE_EXTENSIONS[i]
            exts += "{} Files (*.{});;".format(ext.upper(), ext)
        exts += "All Files (*)"
        return exts

    # Abstract methods
    def create_layers(self):
        """
        Creates each layer of the model.
        """
        raise NotImplementedError("Please implement this method.")

    def learn(self):
        """
        Compiles and fits a model, evaluation is optional.
        """
        raise NotImplementedError("Please implement this method.")

    def predict_output(self):
        """
        Predicts an output for a given list of files/datas.
        """
        raise NotImplementedError("Please implement this method.")

    def load_files_to_predict(self, files):
        """
        Loads files to predict.
        """
        self.filenames = files

    def save_model(self, basename="basename", dir="models"):
        """
        Saves a model.
        """
        if not exists(dir):
            makedirs(dir) # Create a new directory if it doesn't exist

        architecture_file_path = basename + '.json'
        print('\t - Architecture of the neural network: ' + architecture_file_path)

        with open(join(dir, architecture_file_path), 'wt') as json_file:
            architecture = self._model.to_json()
            json_file.write(architecture)

        weights_file_path = join(dir, basename + '.hdf5')
        print('\t - Weights of synaptic connections: ' + weights_file_path)
        self._model.save(weights_file_path)

    def open_model(self, architecture_file_name, weights_file_name):
        """
        Opens an existing model.
        """
        if not exists(architecture_file_name):
            print("ERROR: " + architecture_file_name + " doesn't exist.")
            return
        elif architecture_file_name[-4:] != "json":
            print("ERROR: architecture file extension MUST BE json.")
            return
        
        if not exists(weights_file_name):
            print("ERROR: " + weights_file_name + " doesn't exist.")
            return
        elif weights_file_name[-4:] != "hdf5":
            print("ERROR: weights file extension MUST BE hdf5.")
            return

        json_file = open(architecture_file_name)
        architecture = json_file.read()
        json_file.close()
        # Create a model from a json file
        self._model = model_from_json(architecture)
        # Load weights
        self._model.load_weights(weights_file_name)

    @staticmethod
    def conv2d(inputs, filters, kernel_size=(3, 3), action=None, pool_size=(2, 2), up_size=(2, 2), concat_layer=None):
        """
        Creates and returns a layer with multiple convolution, dropout, up-sampling, max-pooling, etc layers.

        TODO: finish this method.
        """
        if action == "upSampling":
            # MxN Up Sampling
            up     = UpSampling2D(up_size)(inputs)
            # Concatenation
            concat = Concatenate(axis=3)([up, concat_layer])
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

        if action == "maxPooling":
            # MxN Max Pooling
            pool   = MaxPooling2D(pool_size=pool_size)(act)

    @staticmethod
    def _dice_coef(y_true, y_pred, smooth=1):
        """
        From: https://github.com/keras-team/keras/issues/3611 and https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
        """
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    @staticmethod
    def _dice_coef_loss(y_true, y_pred):
        """
        From: https://github.com/keras-team/keras/issues/3611 and https://github.com/keras-team/keras/issues/3611#issuecomment-243108708
        """
        return 1 - NNModel._dice_coef(y_true, y_pred)

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
