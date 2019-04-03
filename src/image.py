try:
    from skimage.transform import rotate
    from skimage.util import random_noise
    import numpy as np
    from random import uniform, choice, randint
    from keras.preprocessing.image import array_to_img
except ImportError as err:
    exit(err)

def transfromXY(x, y):
    """
    Applies random transformations on x and y.
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

    # Zoom (or not) on both images
    zoomOrNot = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[4]])]
    x, y = zoomOrNot(x, y)

    return x, y

def _returnXY(x, y):
    """
    Returns x and y.
    """
    return x, y

def _flipXY(x, y):
    """
    Flips x and y.
    """
    return np.fliplr(x), np.fliplr(y)

def _randomRotationXY(x, y, rot=10):
    """
    Does a random rotation to x and y.
    """
    # Pick a random degree of rotation between 25% on the left and 25% on the right
    rand_deg = uniform(-rot, rot)
    # IMPORTANT: set parameter "preserve_range" to True to have values in [0;255] instead of [0;1].
    x, y = rotate(x, rand_deg, preserve_range=True), rotate(y, rand_deg, preserve_range=True)
    return x, y

def _randomNoiseXY(x, y):
    """
    Adds random noise to x but nothing to y.
    """
    nx = random_noise(x)*255.0
    if np.mean(nx) > np.mean(x)+10:
        # Return x and y because random_noise returned a white image.
        return x, y

    return nx, y

def _randomZoomXY(x, y):
    """
    Zooms randomly on x and y. The same zoom is applied on both images.
    """
    random_zoom = randint(0, int(min(x.shape[:2] + y.shape[:2])/4))
    x_zoomed = x[random_zoom:x.shape[0]-random_zoom, random_zoom:x.shape[1]-random_zoom, :]
    y_zoomed = y[random_zoom:y.shape[0]-random_zoom, random_zoom:y.shape[1]-random_zoom, :]
    x = np.array(array_to_img(x_zoomed).resize(x.shape[:2][::-1]))
    y = np.array(array_to_img(y_zoomed).resize(y.shape[:2][::-1]))
    return x, y

DATA_AUGMENTATION_FUNCTION = {
    # No transformation
    'returnXY'         : _returnXY,
    # Transformations
    'flipXY'           : _flipXY,
    'randomRotationXY' : _randomRotationXY,
    'randomNoiseXY'    : _randomNoiseXY,
    'randomZoomXY'     : _randomZoomXY
}

if __name__ == "__main__":
    # Imports for testing
    try:
        from PIL import Image
        from os.path import exists, join
        from os import mkdir
        from scipy.misc import imsave
    except ImportError as err:
        exit(err)

    # Main directory of the images
    dir_path   = "C:/Users/e_sgouge/Documents/Etienne/Python/analyze_images"
    filename   = "route_2.jpg"
    file       = join(dir_path, filename)
    output_dir = join(dir_path, "tests")

    # Check if file and directory exist
    assert exists(dir_path) == True
    assert exists(file)     == True

    if not exists(output_dir):
        mkdir(output_dir)

    # Open the image
    img = Image.open(file)
    # Transform it to a numpy array
    x, y = np.array(img), np.array(img)

    # FLIP
    x_flip, y_flip = _flipXY(x, y)

    imsave(join(output_dir, "flip_x_" + filename), x_flip)
    imsave(join(output_dir, "flip_y_" + filename), y_flip)

    # ROTATE
    x_rot, y_rot = _randomRotationXY(x, y)

    imsave(join(output_dir, "rot_x_" + filename), x_rot)
    imsave(join(output_dir, "rot_y_" + filename), y_rot)

    # NOISE
    x_noise, _ = _randomNoiseXY(x, y)

    imsave(join(output_dir, "noise_x_" + filename), x_noise)

    # ZOOM
    x_zoom, y_zoom = _randomZoomXY(x, y)

    imsave(join(output_dir, "zoom_x_" + filename), x_zoom)
    imsave(join(output_dir, "zoom_y_" + filename), y_zoom)