#from shutil import copyfile
from os import rename
from os.path import join, exists

if __name__ == "__main__":
    path = "C:/Users/e_sgouge/Documents/Etienne/Python/Reconnaissance_chiffre/datas/data_road/training/x"
    for i in range(96):
        filetocopy = join(path, "umm_" + "0"*(6-(len(str(i)))) + "{}.png".format(i))
        if not exists(filetocopy):
            continue
    
        assert exists(filetocopy) == True

        #filetocreate = join(path, "um_lane_" + "0"*(6-(len(str(i)))) + "{}.png".format(i))
        #copyfile(filetocopy, filetocreate)
        newname = join(path, "umm_road_" + "0"*(6-(len(str(i)))) + "{}.png".format(i))
        if exists(newname):
            continue
        rename(filetocopy, newname)
    


