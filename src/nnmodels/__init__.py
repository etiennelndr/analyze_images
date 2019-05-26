from .model                 import NNModel
from .aerialbuildingsmodel  import AerialBuildingsModel
from .aerialroadsmodel      import AerialRoadsModel
from .animalsmodel          import AnimalsModel
from .digitsmodel           import DigitsModel
from .roadsmodel            import RoadsModel
from .kidneysmodel          import KidneysModel

# This list contains all of the available neural network models
available_models = {
    AerialBuildingsModel.__name__.lower() : AerialBuildingsModel(),
    AerialRoadsModel.__name__.lower() :     AerialRoadsModel(),
    AnimalsModel.__name__.lower() :         AnimalsModel(),
    DigitsModel.__name__.lower() :          DigitsModel(),
    RoadsModel.__name__.lower() :           RoadsModel(),
    KidneysModel.__name__.lower() :         KidneysModel(),
}
