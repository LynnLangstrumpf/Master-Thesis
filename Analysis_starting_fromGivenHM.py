#copyright Pauline Löffler
#----------------------------------------------------------------
'''
This script is designed to read in an averaged HM and analyses
this on the underlying diffusion behaviour. Be sure to recheck the correct
settings correspondent to the acquisition settings in order to obtain reliable
results.
'''


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit


def findData():
    """
    locates and lists all Lif files in the project folder within 'data'
    :return: list of file paths
    """
    data_dir = Path('./data')
    paths_LS = list(data_dir.glob('*.npy'))
    paths_LS.sort()  # by date as my file names always start with the date
    return paths_LS[0]


def Generate_timeline(LS):
    """
    Generates a time-line next to the measured intensities
    :param LS_oneDimension:
    :return: Matrix (t,I(t))
    """
    global LineTime
    LS_tx = []
    for timepoint in range(LS.shape[0]):
        LS_tx.append(
            timepoint * LineTime)  # aquisition at 1800Hz means approx. 0,556 millisec per line
    LS_tx = np.c_[
        LS_tx, LS]  # is now a matrix: column1= time in millisek column2= intensities ...
    return LS_tx


def GaussianFit(STICS, timeline):
    """
    takes the 2DFFT result as well as the corresponding tau values and does several things:
    1) Plots the profil over x for small taus
    2) fits a gaussian (plus a constant) to each line over x
    3) Reads out Sigma and calculates MSD
    4) Takes every 25th line for a 3D plot representation of the data
    5) Logarithmically Bins and averages the MSD values over tau
    6) Exports those to csv for further analysis in prism
    :param STICS:
    :param timeline:
    :return: None
    """
    global max_tau_Gaussian_in_ms
    global LineTime
    global Pixellength
    global identifier

    maxTimelineIndex = int(max_tau_Gaussian_in_ms / LineTime)
    Tau_MSD = []
    STICSMSD = np.flipud(STICS)
    x = np.array(range(STICS.shape[1]))
    plt.plot(x, STICSMSD[0, :])
    plt.show()
    plt.clf()
    x = np.array(range(len(STICS[0, :])))
    x = x.astype(float)
    timeline = timeline[:maxTimelineIndex]
    BestFitList = []
    tauaxis = []
    Tau_mean_previous = 0
    #--------------------------------------------------------------------------
    #fitting over every tau by first getting a guess due to individual constant Gaussian fitting and than fitting the combined model
    #--------------------------------------------------------------------------
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
        params = dict(result.values)
        sigma = params['sigma']
        MSD = 2*((sigma * Pixellength )** 2)
        Tau_MSD.append([tau, MSD])
        if tauindex <= 1000:
            if tauindex % 25 == 0:
                BestFitList.append([x, result.best_fit])
                tauaxis.append(tau)
        if tau < 1:
            plt.plot(x, result.best_fit)
            fig1 = plt.gcf()
            fig1.savefig('Export/' + identifier + '_Gaussianfirst.jpeg', dpi=300)
            plt.show()
            plt.clf()
            plt.plot(x, result.best_fit)
            plt.plot(x, result.residual)
            fig1 = plt.gcf()
            fig1.savefig('Export/' + identifier + '_Gaussianresidual.jpeg',
                         dpi=300)
            plt.show()
            plt.clf()
        print(tauindex)
    fig = plt.figure()
    #--------------------------------------------------------------------------
    #Generating the 3D graph
    #--------------------------------------------------------------------------
    ax = fig.add_subplot(111, projection='3d')
    zs = 0
    for gausfit in BestFitList:
        xs = np.array(gausfit[0])
        ys = np.array(gausfit[1])
        zValues = np.array(tauaxis)
        ax.plot(xs=xs, ys=ys, zs=zValues[zs])
        zs = zs+1
    ax.view_init(elev=100, azim=270)
    ax.set_xlabel('X')
    ax.set_ylabel('AC')
    ax.set_zlabel('tau')
    fig1 = plt.gcf()
    fig1.savefig('Export/'+identifier + '_Gaussians3D.jpeg', dpi=300)
    plt.show()
    #--------------------------------------------------------------------------
    #Averaging the MSD in log-bins, setting the units to s and µm² and exporting the MSD curve as csv
    #--------------------------------------------------------------------------
    Tau_MSD = np.array(Tau_MSD)
    PlotLog = int(np.log10(BinSeconds * 1000))
    Bins = np.logspace(-1, PlotLog, num=100, endpoint=False, base=10)
    StartIndex = 0
    Endindex = 0
    MSDMean = []
    for Bin in Bins:
        for MSDindex, tau in enumerate(Tau_MSD[:, 0]):
            if tau >= Bin:
                Endindex = MSDindex
                break
        # rescaling tau to seconds and MSD to µm² as these are said to be the more commonly used
        Tau_mean = np.mean(Tau_MSD[StartIndex:Endindex + 1, 0])
        Tau_mean = Tau_mean/1000 #ms to s
        MSD_mean = np.mean(Tau_MSD[StartIndex:Endindex + 1, 1])
        MSD_mean = MSD_mean/1000/1000 #nm² to µm²
        MSD_SD = np.std(Tau_MSD[StartIndex:Endindex + 1, 1])
        MSD_SD = MSD_SD / 1000 / 1000  # nm² to µm²
        if not Tau_mean_previous == Tau_mean:
            MSDMean.append([Tau_mean, MSD_mean, MSD_SD])
        Tau_mean_previous = Tau_mean
        StartIndex = Endindex
    Tau_MSD = np.array(MSDMean)
    Tau_MSD_Ex = pd.DataFrame(Tau_MSD, columns=['tau', 'MSD_nm^2','SD'])
    Tau_MSD_Ex.to_csv(path_or_buf='Export/'+identifier + '_TauMSD.csv', header=['tau', 'MSD_µm^2', 'SD'], sep=';', index=False)
    plt.errorbar(Tau_MSD[:, 0], Tau_MSD[:, 1])
    plt.show()
    return


def main():
    global Hz_Aquisition
    global LineTime
    global BinSeconds
    global Pixellength
    global max_tau_Gaussian_in_ms
    global identifier
#--------------------------------------------------------------------------
#those settings should be included in the GUI and need to be checked by the user
#--------------------------------------------------------------------------
    max_tau_Gaussian_in_ms = 1000
    BinSeconds = 10
    Hz_Aquisition = 1800
    LineTime = 1000 / Hz_Aquisition  # in millisec
    Pixellength = 50  # nm
#--------------------------------------------------------------------------
# start of the analysis
#--------------------------------------------------------------------------
    identifier = input('Give an identifier for flagging the data exports')
    file = open('AnalyzedData.csv', 'a')
    file.write('Analysis File Identifier:' + identifier + '\n')
    file.close()
    path = findData()
#--------------------------------------------------------------------------
# Gaussian fitting and MSD extraction
#--------------------------------------------------------------------------
    STICGaus = np.load(path)
    timeline = Generate_timeline(STICGaus)
    timeline = timeline[:,0]
    GaussianFit(STICGaus, timeline)


if __name__ == "__main__":
    main()
