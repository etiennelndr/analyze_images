try:
    from nnmodels import NNModel

    import nibabel as nib
    import numpy as np
except ImportError as err:
    exit(err)

class KidneysModel(NNModel):
    """
    Neural network model for kidney segmentations.

    Sources:
        - kidney MRI and segmentations: https://github.com/neheller/kits19
        In the segmentation, a value of 0 represents background, 1 represents 
        kidney, and 2 represents tumor.
    """

    def __init__(self):
        """
        Initialization of the model.
        """
        super().__init__('model', 'kidneys', model_name=self.__class__.__name__.lower())

        # Input data shape
        self.input_shape = (611, 512, 512) # 512 slices of 611x512 images

        # File extensions for data to predict
        self.FILE_EXTENSIONS  = [
            "nii.gz",
            "nii",
            "hdr"
        ]

    def create_layers(self):
        """
        Creates each layer of the model.
        """
        return super().create_layers()

    def learn(self):
        """
        Compiles and fits a model, evaluation is optional.
        """
        return super().learn()

    def loadDataToPredict(self, filename):
        """
        Loads data to predict.
        """
        # Store the file name
        self.__filename = filename
        # Open the desired picture
        irm = nib.load(filename)
        print(irm.affine)
        # Get the image array
        self.__imgToPredict = irm.get_fdata()
        # Be careful -> each pixel value must be a float
        self.__imgToPredict = self.__imgToPredict.astype("float32")

        print(self.__imgToPredict.shape)
        # Normalize the image
        #self.__imgToPredict /= 255                          # TODO: change this value

    def predictValue(self):
        """
        Predicts a value with a given data.
        """
        return super().predictValue()