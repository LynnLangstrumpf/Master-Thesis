from pathlib import Path
import skimage.io as skio  # read images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from numba import jit
import scipy.fft
import scipy
import lmfit


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
    data_dir = Path('./data')
    paths_LS = list(data_dir.glob('*.npy'))
    paths_LS.sort()  # by date as my file names always start with the date
    return paths_LS



def main():
    paths = findData()
    hmap1 = np.load(paths[0], allow_pickle=True)
    #hmap2 = np.load(paths[1], allow_pickle=True)
    #both STICS are loaded, both go up to 10 sek. Now I need to bin mine to 46 log steps

    paolosHM =pd.read_csv('data/STICS_Paolo.csv', sep=',',header=None)
    # Drop first row
    paolosHM = np.array(paolosHM)
    hmapSTIC = paolosHM[:, 40:104]
    timeline = paolosHM[2:66, -1:]
    newtime = []
    for time in timeline:
        time = str(time)
        start = time.find('[') + 2
        end = time.find(']')
        tau = float(time[start:end-1])
        newtime.append(tau)
    timeline = newtime
    hmapSTIC = hmapSTIC[2:202, :]
    hmapSTIC = np.rot90(hmapSTIC)
    np.save('Export/Paolos_HM.npy', hmapSTIC)
    plt.clf()
    cs = plt.contourf(np.flipud(hmapSTIC), cmap='inferno', levels=100)
    cs.axes.set_xlabel('x position in pixels [a 50 nm]')
    cs.axes.set_ylabel('tau [ms]')
    xBorder = hmapSTIC.shape[1]
    xBorderh = int(xBorder / 2)
    xticks = range(-xBorderh, xBorderh, 10)
    xtickpos = range(0, xBorder, 10)
    cs.axes.set_xticks(xtickpos)
    cs.axes.set_xticklabels(xticks)
    cs.axes.set_yticks(range(hmapSTIC.shape[0]))
    cs.axes.set_yticklabels(timeline[:])
    #plt.ylim(1, )
    #plt.yscale('log')
    fig1 = plt.gcf()
    fig1.savefig('Export/Paolo_STICS_HM.jpeg', dpi=300)
    plt.show()
    #--------cut at 10sec
    hmapSTIC = np.flipud(hmapSTIC)
    hmapSTIC10 = hmapSTIC[:46, :]
    hmapSTIC10 = np.flipud(hmapSTIC10)
    timeline10 = timeline[:46]
    middle = round(hmapSTIC10.shape[1] / 2)
    xlower = middle - 45
    xupper = middle + 45
    hmapSTIC10 = hmapSTIC10[:, xlower:xupper]
    np.save('Export/Paolos_HM_10sek.npy', hmapSTIC10)
    plt.clf()
    cs = plt.contourf(np.flipud(hmapSTIC10), cmap='inferno', levels=100)
    cs.axes.set_xlabel('x position [µm]')
    cs.axes.set_ylabel('tau [s]')
    xBorder = hmapSTIC10.shape[1]
    xBorderh = int(xBorder / 2)
    xticks = []
    for xtick in range(-xBorderh, xBorderh, 10):
        xtick = xtick * 50 / 1000  # converting to µm as this is wanted as standard
        xticks.append(xtick)
    xtickpos = range(0, xBorder, 10)
    cs.axes.set_xticks(xtickpos)
    cs.axes.set_xticklabels(xticks)
    cs.axes.set_yticks([1, 13, 24, 34, 45])
    cs.axes.set_yticklabels([0.001, 0.01, 0.1, 1, 10])
    plt.ylim(1, )
    plt.yscale('log')
    fig1 = plt.gcf()
    fig1.savefig('Export/Paolo_STICS_HM_10sek.jpeg', dpi=300)
    plt.show()
    ACC = hmapSTIC[:,middle:middle+1]
    ACCtimeline = timeline[:-1]
#now I need to implement paolos binning onto my data using his mean as my boaders is imprecise but,...



    #-------my GaussianFit on Paolos HM
    Tau_MSD = []
    STICSMSD = np.flipud(hmapSTIC10)
    x = np.array(range(hmapSTIC10.shape[1]))
    plt.plot(x, STICSMSD[0, :])
    plt.show()
    plt.clf()
    x = np.array(range(len(hmapSTIC10[0, :])))
    x = x.astype(float)
    timeline = np.array(timeline10[:])
    BestFitList = []
    tauaxis = []
    Tau_mean_previous = 0
    for tauindex, tau in enumerate(timeline):
        y = np.array(STICSMSD[tauindex, :])
        y = y.astype(float)
        y = y - np.min(y)
        y = y / np.max(y)
        tau = tau.astype(float)
        plt.plot(x, y)
        cmodel = lmfit.models.ConstantModel()
        cpars = cmodel.guess(y, x=x)
        cresult = cmodel.fit(y, cpars, x=x)
        cparams = dict(cresult.values)
        c = cparams['c']
        gmodel = lmfit.models.GaussianModel()
        gpars = gmodel.guess(y, x=x)
        gresult = gmodel.fit(y, gpars, x=x)
        gparams = dict(gresult.values)
        amp = gparams['amplitude']
        cen = gparams['center']
        sig = gparams['sigma']
        cgmodel = lmfit.models.ConstantModel() + lmfit.models.GaussianModel()
        result = cgmodel.fit(y, x=x, c=c, amplitude=amp, center=cen, sigma=sig)
        #print(result.fit_report())
        params = dict(result.values)
        sigma = params['sigma']
        MSD = (sigma * 50 * 2) ** 2
        MSD = MSD /1000 /1000 #convert to µm²
        Tau_MSD.append([tau, MSD])
        BestFitList.append([x, result.best_fit])#here all because all ready binned
        Y_fitted = np.array(result.best_fit)
        Y_rest = y-Y_fitted
        tauaxis.append(tau)
        if tau < 1:
            plt.plot(x, Y_rest)
            plt.plot(x,result.best_fit)
           # fig1 = plt.gcf()
            #fig1.savefig('Export/'+str(tau)+'.jpeg', dpi=300)
           # plt.show()
            plt.clf()
        print(tauindex)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zs = 0
    for gausfit in BestFitList:
        xs = np.array(gausfit[0])
        ys = np.array(gausfit[1])
        zValues = np.array(tauaxis)
        ax.plot(xs=xs, ys=ys, zs=zValues[zs])
        zs = zs + 1
    ax.view_init(elev=100, azim=270)
    ax.set_xlabel('X')
    ax.set_ylabel('AC')
    ax.set_zlabel('tau')
    fig1 = plt.gcf()
    fig1.savefig('Export/PaoloHM10sek_Gaussians3D.jpeg', dpi=300)
    plt.show()
    Tau_MSD = np.array(Tau_MSD)
    Tau_MSD_Ex = pd.DataFrame(Tau_MSD, columns=['tau', 'MSD_µm^2'])
    Tau_MSD_Ex.to_csv(path_or_buf='Export/Paolo10sek_TauMSD.csv',
                      header=['tau', 'MSD_nm^2'], sep=';', index=False)
    plt.errorbar(Tau_MSD[:, 0], Tau_MSD[:, 1])
    plt.show()
    ACCAnalyse = []
    timeline = ACCtimeline
    for index,tau in enumerate(timeline):
        ACValue = float(ACC[index])
        ACCAnalyse.append([tau, ACValue])
    ACC = np.array(ACCAnalyse)
    ACC = ACC[:52, :]
    xNullEx = pd.DataFrame(ACC,
                           columns=['tau', 'G_tau_Average'])
    xNullEx.to_csv(path_or_buf='Export/Paolo_xNullEx.csv',
                   header=['tau', 'AC_Average'], sep=';', index=False)
    print(ACC)
    FitACFunctionGetD(ACC,1)

if __name__ == "__main__":
    main()