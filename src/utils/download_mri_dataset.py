try:
    import requests
    import sys
    import zipfile
    import pandas as pd
    import numpy as np
    import os
    import nibabel
    import matplotlib.pyplot as plt
except ImportError as err:
    exit(err)

dir = "./fMRI_data"

if os.path.exists(dir) ==  False:
    os.mkdir(dir.split("./")[0])
else:
    if os.path.exists(dir + "/sM00223") and os.path.exists(dir + "/fM00223"):
        print("You have already extracted these datas.")
        sys.exit()

# Define the URL of the data and download it using the Requests libary
url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/MoAEpilot.zip'
data = requests.get(url)

open(dir + "/data.zip", "wb").write(data.content)

# Unzip the file
zip_ref = zipfile.ZipFile(dir + "/data.zip", "r")
zip_ref.extractall(dir)
zip_ref.close()
