from os import rename, listdir
from os.path import exists, isfile, join
from PIL import Image
import numpy as np

#dir_path = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre/datas/data_road/validation/y"
dir_path = "D:/Documents/Programmation/Python/analyze_images/datas/data_road/training/y"

assert exists(dir_path) == True

files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]

print(files)

for f in files:
    img = Image.open(f)
    img_array = np.array(img)
    img_array[img_array[:,:,2] > 200] = [0,255,0]

    # Create a new Image instance with the new_img_array array
    new_img = Image.fromarray(img_array.astype('uint8'))
    # Finally, save this image
    new_img.save(f)
    print("OK", f)