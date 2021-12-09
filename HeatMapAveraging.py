"""
NOTE: The way this is implemented requires the user to be sure the data was acquired with identical Pixel-size and Hz Acquisition speed as I do not check or correct for it!
As the HMs are of different x sizes, depending on the chosen x interval in the analysis --> the averaged HM will be cropped to fit the smallest HM
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def findData():
    """
    locates and lists all Lif files in the project folder within 'data'
    :return: list of file paths
    """
    data_dir = Path('./data_Average')
    paths_LS = list(data_dir.glob('*.npy'))
    paths_LS.sort()  # by date as my file names always start with the date
    return paths_LS


def loadingArrayAndSortByMatchingTimelines(path_LS):
    """
    Theoretically all time lines should be the same as there is no reason to change that parameter when analyzing one batch of supposedly similar behaviour
    Also the tau Interval steps need to be the same as they depend on acquisition speed. Thus checking for the shape in tau direction should do the trick.
    :return:
    """
    readHMNumber = 0
    totalHMNumber = len(path_LS)
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
    print('The minimal heatmap shape to be used is:',minShapetau, 'x', minShapex)
    readHMNumber = 0
    xminhalf = int(minShapex/2)
    for HM in path_LS:
        hmap = np.load(HM)
        middle = round(hmap.shape[1]/2)
        xlower = middle - xminhalf
        xupper = middle + xminhalf
        hmap = hmap[:minShapetau,xlower:xupper]
        if readHMNumber == 0:
            SummedHMs = hmap
            readHMNumber = readHMNumber + 1
        else:
            SummedHMs = SummedHMs + hmap
            readHMNumber = readHMNumber + 1
    AveragedHM = SummedHMs/readHMNumber
    print(readHMNumber, 'used from a total of ', totalHMNumber, ' Heatmaps.')
    return AveragedHM


def main():
    paths = findData()
    AveragedHM = loadingArrayAndSortByMatchingTimelines(paths)
    np.save('Export/Averaged_HM.npy', AveragedHM)
    cs = plt.contourf(np.flipud(AveragedHM), cmap='inferno', levels=100)
    cs.axes.set_xlabel('x position [µm]')
    cs.axes.set_ylabel('tau [ms]')
    xBorder = AveragedHM.shape[1]
    xBorderh = int(xBorder / 2)
    xticks = []
    for xtick in range(-xBorderh, xBorderh, 10):
        xtick = xtick*50/1000 #converting to µm as this is wanted as standard
        xticks.append(xtick)
    xtickpos = range(0, xBorder, 10)
    cs.axes.set_xticks(xtickpos)
    cs.axes.set_xticklabels(xticks)
    plt.ylim(1, )
    plt.yscale('log')
    fig1 = plt.gcf()
    fig1.savefig('Export/Averaged_HM.jpeg', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()