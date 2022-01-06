#copyright Pauline Löffler
"""
NOTE: The way this is implemented requires the user to be sure the data was acquired with identical pixel-size and Hz Acquisition speed as I do not check or correct for it!
Allows the user to crop the HM to a common size for easy side-by-side comparison
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def findData():
    """
    locates and lists all Lif files in the project folder within 'data'
    :return: list of file paths
    """
    data_dir = Path('./data_Crop')
    paths_LS = list(data_dir.glob('*.npy'))
    paths_LS.sort()  # by date as my file names always start with the date
    return paths_LS


def loadingArrayAndCroppingToCommonSize(path_LS):
    """
    Checks for minimal HM shape and crops and exports all HM accordingly as JPG and npy
    :return: None
    """
    readHMNumber = 0
    for HM in path_LS:
        hmap = np.load(HM)
        if readHMNumber == 0:
            minShapetau = hmap.shape[0]
            minShapex = hmap.shape[1]
            readHMNumber = readHMNumber + 1
        else:
            if hmap.shape[0] < minShapetau:
                minShapetau = hmap.shape[0]
            if hmap.shape[1] < minShapex:
                minShapex = hmap.shape[1]
            readHMNumber = readHMNumber + 1
    print('The smallest heatmap shape to be used is:',minShapetau, ' in ms and x', minShapex, ' in pixels[a 50 nm]')
    xuser = input('Give x limit in pixels or press ENTER if you want to use the biggest possible') or 0
    xuser = int(xuser)
    if not xuser == 0:
        if xuser > minShapex:
            print('Given x limit is to big. Set x limit to',minShapex, ' in pixels[a 50 nm]')
        else: minShapex = xuser
    xminhalf = int(minShapex/2)
    for HM in path_LS:
        hmap = np.load(HM)
        HM = str(HM) #extracting the name
        HM = HM.replace('data_Crop\\', '')
        HM = HM.replace('.npy', '')
        middle = round(hmap.shape[1]/2)
        xlower = middle - xminhalf
        xupper = middle + xminhalf
        hmap = hmap[:minShapetau,xlower:xupper] #cutting it to the chosen size
        np.save('Export/'+HM+'cropped.npy', hmap) #saving it as 'cropped'
        cs = plt.contourf(np.flipud(hmap), cmap='inferno', levels=100)
        cs.axes.set_xlabel('x position [µm]')
        cs.axes.set_ylabel('tau [ms]')
        xBorder = hmap.shape[1]
        xBorderh = int(xBorder / 2)
        xticks = []
        for xtick in range(-xBorderh, xBorderh, 10):
            xtick = xtick * 50 / 1000  # converting to µm
            xticks.append(xtick)
        xtickpos = range(0, xBorder, 10)
        cs.axes.set_xticks(xtickpos)
        cs.axes.set_xticklabels(xticks)
        plt.ylim(1, )
        plt.yscale('log')
        fig1 = plt.gcf()
        fig1.savefig('Export/'+HM+'cropped.jpeg', dpi=300)
        print(HM,'processed')
        plt.clf()
    return


def main():
    paths = findData()
    loadingArrayAndCroppingToCommonSize(paths)


if __name__ == "__main__":
    main()
