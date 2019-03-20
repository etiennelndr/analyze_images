try:
    from os import makedirs
    from shutil import copyfile
    from os.path import join, exists
except ImportError as err:
    exit(err)

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre/datas/dogs-vs-cats/train"

# The directory where we will
# store our smaller dataset
base_dir = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre/datas/dogs_vs_cats"
makedirs(base_dir, exist_ok=True)

# Directories for our training, validation 
# and test splits
# Train
train_dir = join(base_dir, "train")
makedirs(train_dir, exist_ok=True)
# Validation
validation_dir = join(base_dir, "validation")
makedirs(validation_dir, exist_ok=True)
# Test
test_dir = join(base_dir, "test")
makedirs(test_dir, exist_ok=True)

# TRAINING
# Directory with our training cat pictures
train_cats_dir = join(train_dir, 'cats')
makedirs(train_cats_dir, exist_ok=True)
# Directory with our training dog pictures
train_dogs_dir = join(train_dir, 'dogs')
makedirs(train_dogs_dir, exist_ok=True)

# VALIDATION
# Directory with our validation cat pictures
validation_cats_dir = join(validation_dir, 'cats')
makedirs(validation_cats_dir, exist_ok=True)
# Directory with our validation dog pictures
validation_dogs_dir = join(validation_dir, 'dogs')
makedirs(validation_dogs_dir, exist_ok=True)

# TEST
# Directory with our validation cat pictures
test_cats_dir = join(test_dir, 'cats')
makedirs(test_cats_dir, exist_ok=True)

# Directory with our validation dog pictures
test_dogs_dir = join(test_dir, 'dogs')
makedirs(test_dogs_dir, exist_ok=True)

def copyFiles(filename, dir, start, stop):
    global original_dataset_dir
    fnames = [filename.format(i) for i in range(start, stop)]
    for fname in fnames:
        src = join(original_dataset_dir, fname)
        dst = join(dir, fname)
        if not exists(dst):
            copyfile(src, dst)

# CATS
# Copy first 1000 cat images to train_cats_dir
copyFiles('cat.{}.jpg', train_cats_dir, 0, 1000)

# Copy next 500 cat images to validation_cats_dir
copyFiles('cat.{}.jpg', validation_cats_dir, 1000, 1500)

# Copy next 500 cat images to test_cats_dir
copyFiles('cat.{}.jpg', test_cats_dir, 1500, 2000)

# DOGS
# Copy first 1000 cat images to train_dogs_dir
copyFiles('dog.{}.jpg', train_dogs_dir, 0, 1000)

# Copy next 500 cat images to validation_dogs_dir
copyFiles('dog.{}.jpg', validation_dogs_dir, 1000, 1500)

# Copy next 500 cat images to test_dogs_dir
copyFiles('dog.{}.jpg', test_dogs_dir, 1500, 2000)
