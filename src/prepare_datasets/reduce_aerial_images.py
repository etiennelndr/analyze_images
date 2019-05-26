try:
    from os.path import join, exists, isfile
    from os import listdir, remove

    from PIL import Image

    import numpy as np

    from shutil import copyfile

    from scipy.misc import imsave
except ImportError as err:
    exit(err)

if __name__ == "__main__":
    # Main folder
    main_folder = "C:/Users/e_sgouge/Documents/Etienne/Python/analyze_images/datas/aerial_roads"
    # Training
    training_dir = join(main_folder, "training")
    training_x_dir = join(training_dir, "x")
    training_y_dir = join(training_dir, "y")
    # Validation
    validation_dir = join(main_folder, "validation")
    validation_x_dir = join(validation_dir, "x")
    validation_y_dir = join(validation_dir, "y")
    # Testing
    testing_dir = join(main_folder, "testing")
    testing_x_dir = join(testing_dir, "x")
    testing_y_dir = join(testing_dir, "y")

    folders = [
        # Training
        training_x_dir,
        training_y_dir,
        # Validation
        validation_x_dir,
        validation_y_dir,
        # Testing
        testing_x_dir,
        testing_y_dir
    ]

    nbrOfFiles = [
        # Training
        1108,
        1108,
        # Validation
        14,
        14,
        # Testing
        49,
        49
    ]

    for f in range(len(folders)):
        folder = folders[f]

        assert exists(folder) == True

        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

        print(len(files))
        print(files)
    
        if len(files) == 9*nbrOfFiles[f]:
            print("continue")
            continue

        for file in files:
            img = Image.open(file)
            img_array = np.array(img)

            assert img_array.shape[0]%500 == 0
            assert img_array.shape[1]%500 == 0

            if img_array.shape[0] == 500 and img_array.shape[1] == 500:
                continue

            for i in range(int(img_array.shape[0]/500)):
                for j in range(int(img_array.shape[1]/500)):
                    if len(img_array.shape) == 3:
                        sub_array = img_array[i*500:(i+1)*500:, j*500:(j+1)*500:, :]
                    elif len(img_array.shape) == 2:
                        sub_array = img_array[i*500:(i+1)*500:, j*500:(j+1)*500:]
                    else:
                        raise NotImplementedError("Unknown shape: {}".format(img_array.shape))

                    subfile_name = file.split(".")[0] + "_{}{}.".format(i,j) + file.split(".")[1]
                    if not exists(subfile_name):
                        imsave(subfile_name, sub_array)
        
            # Close the pointer
            img.close()
            # Delete the old file
            remove(join(folder, file))