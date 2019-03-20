from os import rename, listdir
from os.path import exists, isfile, join
from PIL import Image
import numpy as np

#dir_path = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre/datas/data_road/validation/y"
dir_path = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre/datas/data_road/training/y"

assert exists(dir_path) == True

files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]

print(files)

for f in files:
    img = Image.open(f)
    img_array = np.array(img)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            #if img_array[i,j,:][0] == 255 and img_array[i,j,:][2] == 255:
            #    img_array[i,j,:] = np.array([0,255,0])
            if np.array_equal(img_array[i,j,:], np.array([0,0,0])):
                img_array[i,j,:][1] = 255

    # Create a new Image instance with the new_img_array array
    new_img = Image.fromarray(img_array.astype('uint8'))
    # Finally, save this image
    new_img.save(f)
    print("OK", f)