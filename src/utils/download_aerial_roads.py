try:
    from urllib.request import urlretrieve, urlopen
    from os.path import join, exists
except ImportError as err:
    exit(err)

# Training
training_x_url   = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/"
training_y_url   = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/"
# Validation
validation_x_url = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/"
validation_y_url = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/"
# Testing
testing_x_url    = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/"
testing_y_url    = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/"

urls = [
    # Training
    training_x_url,
    training_y_url,
    # Validation
    validation_x_url,
    validation_y_url,
    # Testing
    testing_x_url,
    testing_y_url
]

#main_folder = "D:/Documents/Programmation/Python/analyze_images/datas/aerial_roads"
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

for f in folders:
    assert exists(f) == True

for i in range(len(urls)):
    response = str(urlopen(urls[i]).read())

    resps = response.split('href="')

    if 'map' in urls[i]:
        resps = [r.split('">')[0] for r in resps if r.split('">')[0][-3:] == "tif"]
    else:
        resps = [r.split('">')[0] for r in resps if r.split('">')[0][-4:] == "tiff"]

    for url in resps:
        print(url)
        filename = url.split("/")[-1]
        filename = join(folders[i], filename)
        if not exists(filename):
            urlretrieve(url, filename)