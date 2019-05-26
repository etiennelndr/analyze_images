try:
    from os import listdir
    import nibabel as nib
    from numpy import rot90, transpose, array
    import matplotlib.pyplot as plt
    from random import randrange
except ImportError as err:
    exit(err)

if __name__ == "main__":
    # Find all files in the structural data folder
    #data_path = "./fMRI_data/fM00223/"
    data_path = "./datas/HCP_MPRAGE/"
    files = listdir(data_path)

    # Basic information about the data acquisition
    #x_size = 64
    #y_size = 64
    #n_slice = 64
    #n_volumes = 96

    # Read in the data
    data_all = []
    #for data_file in files:
    #    if data_file[-3:] == "hdr" or data_file[-3:] == "nii":
    #        data = nib.load(data_path + data_file).get_data()
    #        print(data.shape)
    #        data_all.append(data)#.reshape(x_size, y_size, n_slice))
    #        break
    i = randrange(0, len(files))
    print(i)
    data = nib.load(data_path + files[i]).get_data()
    data_all.append(data)

    print(array(data_all).shape)

    fig, ax = plt.subplots(3, 6, figsize=[18, 8])

    # Organize the data for visualisation in the coronal plane
    coronal = transpose(data_all, [2, 3, 1, 0])#, [1, 3, 2, 0])
    coronal = rot90(coronal, 0)

    print("coronal     ->", coronal.shape)

    # Organize the data for visualisation in the transversal plane
    transversal = transpose(data_all, [1, 3, 2, 0])#, [2, 1, 3, 0])
    transversal = rot90(transversal, 1)

    print("transversal ->", transversal.shape)

    # Organize the data for visualisation in the sagittal plane
    sagittal = transpose(data_all, [2, 1, 3, 0])#, [2, 3, 1, 0])
    sagittal = rot90(sagittal, 0)

    print("sagittal    ->", sagittal.shape)

    # Index
    index = 0

    # Plot some of the images in different planes
    n = 10
    for i in range(6):
        ax[0][i].imshow(coronal[:, :, n, index], cmap='gray')
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        if i == 0:
            ax[0][i].set_ylabel('coronal', fontsize=25, color='r')
        n += 42
    
    n = 5
    for i in range(6):
        ax[1][i].imshow(transversal[:, :, n, index], cmap='gray')
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        if i == 0:
            ax[1][i].set_ylabel('transversal', fontsize=25, color='r')
        n += 42
    
    n = 5
    for i in range(6):
        ax[2][i].imshow(sagittal[:, :, n, index], cmap='gray')
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        if i == 0:
            ax[2][i].set_ylabel('sagittal', fontsize=25, color='r')
        n += 28

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
