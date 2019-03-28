try:
    from .model import NNModel
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
        self.FILE_EXTENSIONS  = [
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ]
