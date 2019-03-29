try:
    import requests
    import zipfile
    from os import mkdir
    from os.path import exists
except ImportError as err:
    exit(err)

if __name__ == "__main__":
    dir = "./fMRI_data"

    if path.exists(dir) == False:
        mkdir(dir.split("./")[0])
    else:
        if exists(dir + "/sM00223") and exists(dir + "/fM00223"):
            print("You have already extracted these datas.")
            exit(0)

    # Define the URL of the data and download it using the Requests libary
    url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/MoAEpilot.zip'
    data = requests.get(url)

    open(dir + "/data.zip", "wb").write(data.content)

    # Unzip the file
    zip_ref = zipfile.ZipFile(dir + "/data.zip", "r")
    zip_ref.extractall(dir)
    zip_ref.close()
