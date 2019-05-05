try:
    from skimage.transform import rotate, resize
    from skimage.util import random_noise
    from os import listdir
    from os.path import exists, join, isfile, splitext, basename
    from PIL import Image
    import numpy as np
    from imageio import imwrite
    from random import uniform, choice, randint
    from keras.preprocessing.image import array_to_img
except ImportError as err:
    exit(err)

def change_values_in_array(a, val_old, val_new):
    """
    Given a numpy array, it changes each specific value in another.
    This function is recursive.

    Attributes:
        - a      : the numpy array to work on
        - val_old: old values
        - val_new: new values

    Returns:
        A numpy array of the same shape.

    Example:
        >>> import numpy as np
        >>> arr = np.array([3, 2, 4, 0, 4, 0, 2, 1, 0, 1])
        >>> old = np.array([0, 1, 3])
        >>> new = np.array([29, 42, 13])
        >>> arr = change_values_in_array(a, old, new)
        >>> print(arr)
        [13  2  4 29  4 29  2 42 29 42]

    Adapted from: https://stackoverflow.com/questions/29407945/find-and-replace-multiple-values-in-python (Ashwini_Chaudhary solution)
    """
    try:
        arr = np.arange(a.max()+1, dtype=val_new.dtype)
        arr[val_old] = val_new
        return arr[a]
    except IndexError as err:
        val = int(str(err).split(" is out")[0].split("index ")[1])
        index = np.where(val_old == val)[0][0]
        return change_values_in_array(a, val_old[val_old != val], np.delete(val_new, index))

def transform_shape(shape):
    """
    Transforms a shape to resize an image after that. Shape size must 
    be equal to 2 or 3, otherwise it raises a ValueError.

    Returns:
        A tuple of the same size.

    Raises:
        A ValueError if the shape size is not equal to 2 or 3.

    Example:
        >>> shape = (4850, 650, 3)
        >>> new_shape = transform_shape(shape)
        >>> print(new_shape)
        (5000, 1000, 3)
    """
    if len(shape) <= 1 or len(shape) > 3:
        raise ValueError("ERROR: Shape size must be in [2;3]")
    # Create a tuple to store the new shape
    new_shape = tuple()
    for value in shape[:2]:
        # Convert this value to a string
        val = str(value)
        # Get the first two values and store the rest in another 
        # variable
        sup, inf = val[:-3] if val[:-3]!='' else "1", val[-3:]
        if int(inf) > 500:
            sup = str(int(sup)+1)
        new_shape += (int(sup + "000"),)

    # Don't forget the last element (only if it exists)
    if len(shape) == 3:
        new_shape += (shape[2],)

    return new_shape

def crop_and_resize(dir, result_dir, cropping_size=(1000, 1000), new_size=(336,336)):
    """
    Crop and resize a list of images. 
    """
    assert exists(dir)        == True
    assert exists(result_dir) == True

    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

    for file in files:
        print(file)
        # Get the filename and its extension
        filename, ext = splitext(basename(file))
        # Open the file as an PIL.Image instance
        img = Image.open(file)
        # Convert this Image to a numpy array
        im = np.array(img)
        # Get a new shape
        shape = transform_shape(im.shape)
        # Then resize this image with the new shape
        im = resize(im, shape)
        # For each subarray, create a new image
        for i in range(int(im.shape[0]/cropping_size)):
            for j in range(int(im.shape[1]/cropping_size)):
                if exists(join(result_dir, filename + "_{}{}".format(i,j) + ext)):
                    continue
                # Crop the image in a sub-array
                im_crop = im[i*cropping_size[0]:(i+1)*cropping_size[0], j*cropping_size[1]:(j+1)*cropping_size[1]]
                # Resize it to reduce the shape
                im_crop_resized = resize(im_crop, new_size + im_crop.shape[2:])
                # Finally, save it as a new image
                imwrite(join(result_dir, filename + "_{}{}".format(i,j) + ext), im_crop_resized)

def transfromXY(x, y):
    """
    Applies random transformations on x and y.
    """
    functions = list(DATA_AUGMENTATION_FUNCTIONS)
    # Flip (or not) both images
    flipOrNot = DATA_AUGMENTATION_FUNCTIONS[choice([functions[0], functions[1]])]
    x, y = flipOrNot(x, y)
    # Rotate (or not) both images
    rotateOrNot = DATA_AUGMENTATION_FUNCTIONS[choice([functions[0], functions[2]])]
    x, y = rotateOrNot(x, y)
    # Add noise (or not) to the first image
    #addNoiseOrNot = DATA_AUGMENTATION_FUNCTION[choice([functions[0], functions[3]])]
    #x, y = addNoiseOrNot(x, y)
    # Zoom (or not) on both images
    zoomOrNot = DATA_AUGMENTATION_FUNCTIONS[choice([functions[0], functions[4]])]
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

def _rotationXY(x, y, rot=10, random_rot=True):
    """
    Rotation to x and y.
    If 'random_rot' is set to True it does a random rotation between [rot;rot].
    """
    if random_rot:
        # Pick a random degree of rotation between rot% on the left and rot% on the right
        rand_deg = uniform(-rot, rot)
    else:
        rand_deg = rot
    # IMPORTANT: set parameter "preserve_range" to True to have values in [0;255] instead of [0;1].
    x, y = rotate(x, rand_deg, preserve_range=True), rotate(y, rand_deg, preserve_range=True)
    return x, y

def _randomNoiseXY(x, y):
    """
    Adds random noise to x but nothing to y.
    """
    dtype = x.dtype
    nx = random_noise(x)*255.0
    if np.mean(nx) > np.mean(x)+10:
        # Return x and y because random_noise returned a white image.
        return x, y
    return nx.astype(dtype), y

def _zoomXY(x, y, zoom=None):
    """
    Zooms on x and y. The same zoom is applied on both images.
    If no value is applied to 'zoom' a random value is set.
    """
    if zoom is None:
        zoom = randint(0, int(min(x.shape[:2] + y.shape[:2])/4))
    x_zoomed = x[random_zoom:x.shape[0]-random_zoom, random_zoom:x.shape[1]-random_zoom, :]
    y_zoomed = y[random_zoom:y.shape[0]-random_zoom, random_zoom:y.shape[1]-random_zoom, :]
    x = np.array(array_to_img(x_zoomed).resize(x.shape[:2][::-1]))
    y = np.array(array_to_img(y_zoomed).resize(y.shape[:2][::-1]))
    return x, y

DATA_AUGMENTATION_FUNCTIONS = {
    # No transformation
    'returnXY'         : _returnXY,
    # Transformations
    'flipXY'           : _flipXY,
    'rotationXY'       : _rotationXY,
    'randomNoiseXY'    : _randomNoiseXY,
    'zoomXY'           : _zoomXY
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
    x = np.array(img)
    y = x.copy()

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