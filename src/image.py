try:
    from skimage.transform import rotate
    from skimage.util import random_noise
    from numpy import fliplr, mean, amax
    from random import uniform, choice
except ImportError as err:
    exit(err)

def transfromXY(x, y):
    """
    Apply a random transformation on x and y.
    """
    functions = list(DATA_AUGMENTATION_FUNCTION)
    # Flip (or not) both images
    flipOrNot = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[1]])]
    x, y = flipOrNot(x, y)
    # Rotate (or not) both images
    rotateOrNot = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[2]])]
    x, y = rotateOrNot(x, y)
    # Add noise (or not) to the first image
    #addNoiseOrNot = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[3]])]
    #x, y = addNoiseOrNot(x, y)

    return x, y

def _returnXY(x, y):
    """
    This method only returns x and y.
    """
    return x, y

def _flipXY(x, y):
    """
    Flip x and y.
    """
    return fliplr(x), fliplr(y)

def _randomRotationXY(x, y, rot=10):
    """
    Random rotation for x and y.
    """
    # Pick a random degree of rotation between 25% on the left and 25% on the right
    rand_deg = uniform(-rot, rot)
    # FIX: set parameter "preserve_range" to True to have values in [0;255] instead of [0;1].
    return rotate(x, rand_deg, preserve_range=True), rotate(y, rand_deg, preserve_range=True)

def _randomNoiseXY(x, y):
    """
    Add random noise x but nothing to y.
    """
    nx = random_noise(x)*255.0
    if mean(nx) > mean(x)+10:
        # Return x and y because random_noise returned a white image.
        return x, y

    return nx, y

DATA_AUGMENTATION_FUNCTION = {
    'returnXY'         : _returnXY,
    'flipXY'           : _flipXY,
    'randomRotationXY' : _randomRotationXY,
    'randomNoiseXY'    : _randomNoiseXY
}

if __name__ == "__main__":
    # Imports for testing
    try:
        import numpy as np
        from PIL import Image
        from os.path import exists, join
        from scipy.misc import imsave
    except ImportError as err:
        exit(err)

    # Main directory of the images
    dir_path = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre"
    filename = "qd3.jpg"
    file     = join(dir_path, filename)

    # Check if file and directory exist
    assert exists(dir_path) == True
    assert exists(file)     == True

    # Open the image
    img = Image.open(file)
    # Transform it to a numpy array
    x, y = np.array(img), np.array(img)

    # FLIP
    x_flip, y_flip = _flipXY(x, y)

    imsave(join(dir_path, "flip_x_" + filename), x_flip)
    imsave(join(dir_path, "flip_y_" + filename), y_flip)

    # ROTATE
    x_rot, y_rot = _randomRotationXY(x, y)

    imsave(join(dir_path, "rot_x_" + filename), x_rot)
    imsave(join(dir_path, "rot_y_" + filename), y_rot)

    # NOISE
    x_noise, _ = _randomNoiseXY(x, y)

    imsave(join(dir_path, "noise_x_" + filename), x_noise)