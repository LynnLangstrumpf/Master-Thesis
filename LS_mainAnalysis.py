#copyright Pauline Löffler
#----------------------------------------------------------------
'''
This script is designed to read in a LS as a one-paged tif file and analyses
this on the underlying diffusion behaviour. Be sure to recheck the correct
settings correspondent to the acquisition settings in order to obtain reliable
results.
'''



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
import argparse

HistogramList = []
ShowPlot = False
DValues = []

def findData():
    """
    locates and lists all Lif files in the project folder within 'data'
    :return: list of file paths
    """
    data_dir = Path('./data')
    paths_LS = list(data_dir.glob('*.tif')) # lists all lifs in this folder
    paths_LS.sort()  # by date as my file names always start with the date. Might be an idea to let it run sequentailly for several LS
    return paths_LS


def readFile():
    """
    reads in the tif file of the LS to a ndarray of pixel intensities
    :return: full Line scan as Matrix240830_Sugar_CS2_TP (space x time)
    """
    paths_LS = findData()
    file = open('AnalyzedData.csv', 'a')
    file.write(str(paths_LS)+'\n')
    file.close()
    LS = skio.imread(paths_LS[0])  # starts with the first tif (could be cast into a loop for processing data batches)
    return LS #TODO hier sollte das Bild in LS geladen sein. es sollte ein 16 bit GS image sein.

def Averaging_Gtau_over_Tau(BinnedSTICS):
    """
    Takes the Autocorrelation Curve at deltaX=0 (comparable 1DFFT all averaged) and bins the data to get a smooth curve to plot as on-the-run control if the analysis works
    :param BinnedSTICS: Autocorrelation Curve at deltaX=0
    :return: binned Autocorrelation curve as array with the colums: mean tau, mean AC value, SD of the mean AC value
    """
    AverageValue = 2
    CalcValue = 2
    StartIndex = 0
    BinnedSTICSMean = []
    while StartIndex < np.shape(BinnedSTICS)[0]:
        Endindex = StartIndex + AverageValue - 1
        if Endindex > np.shape(BinnedSTICS)[0] - 1:
            break
        Tau_mean = np.mean(BinnedSTICS[StartIndex:Endindex + 1, 0])
        G_mean = np.mean(BinnedSTICS[StartIndex:Endindex + 1, 1])
        G_SD = np.std(BinnedSTICS[StartIndex:Endindex + 1, 1])
        BinnedSTICSMean.append([Tau_mean, G_mean, G_SD])
        StartIndex = Endindex + 1
        CalcValue = (CalcValue * np.lib.scimath.sqrt(2))
        AverageValue = round(CalcValue)
    return BinnedSTICSMean


def GeneratePlot_Gt_tau(PlottableSTICS):
    """
    Generates the plot and saves it as svg
    :param PlottableSTICS: Binned ACC as array with the colums: mean tau, mean AC value, SD of the mean AC value
    :return: none, just plot
    """
    global LineTime
    global identifier
    global ShowPlot

    xAxis = []
    yAxis = []
    SD = []
    for tau in PlottableSTICS[:, 0]:
        xAxis.append(tau * LineTime)
    for value in PlottableSTICS[:, 1]:
        yAxis.append(value)
    for SDValue in PlottableSTICS[:, 2]:
        SD.append(SDValue)
    plt.errorbar(xAxis, yAxis, SD)
    plt.title('One-dimensional (t)')
    plt.xlabel('\u03C4 [ms]')
    plt.ylabel('G(\u03C4)')
    plt.grid(True)
    plt.xscale('log')
    if ShowPlot:
        plt.show()
    plt.savefig('Gtau_1dim.svg', dpi=600)
    return


def CalculateSTICS(LS):
    """
    Saves the user input to be reproducible, calculates the STICS and shows it as contour plot. Exports this as picture and nparray.
    :param xstart:Userinput. Boarders for the analysis area
    :param xstop:Userinput. Boarders for the analysis area
    :param LS: the whole LS as Matrix
    :return: Autocorrelation over x=0 and the STICS as matrix
    """
    global Hz_Aquisition
    global LineTime
    global BinSeconds
    global Pixellength
    global identifier
    global ShowPlot
    #--------------------------------------------------------------------------
    #calculation of the STICS function
    #--------------------------------------------------------------------------
    StartTimepoint = 0
    BinCount = 0
    while StartTimepoint < LS.shape[0]:
        EndTimepoint = StartTimepoint + (Hz_Aquisition * BinSeconds)
        if EndTimepoint + 1 > LS.shape[0]:
            break
        else:
            LS_bin = LS[StartTimepoint:EndTimepoint]
            #LS_bin = LS_bin-np.min(LS_bin)
            #LS_bin = LS_bin/np.max(LS_bin)
            b_w = np.array(scipy.fft.rfft2(LS_bin))
            c_w = b_w * np.conj(b_w)
            d_t = scipy.fft.irfft2(c_w)
            d_t = np.real(d_t)
        if StartTimepoint == 0:
            STIC = d_t
        else:
            STIC = STIC + d_t
        BinCount = BinCount + 1
        StartTimepoint = EndTimepoint
        print(StartTimepoint)
    STIC = STIC / BinCount
    print(STIC)
    #--------------------------------------------------------------------------
    # HM graph generation and STICS export
    #--------------------------------------------------------------------------
    hmapSTIC = STIC
    xBorder = int(hmapSTIC.shape[1] / 2)
    tauBorder = int(hmapSTIC.shape[0] / 2)
    rightSTICs = hmapSTIC[:, :xBorder]
    leftSTICs = hmapSTIC[:, xBorder:]
    hmapSTIC = np.c_[leftSTICs, rightSTICs]
    hmapSTIC = hmapSTIC[tauBorder:, :]
    hmapSTICfw = hmapSTIC
    xminhalf = int(40/2) # TODO To be defined further. HARDCODED
    middle = round(hmapSTIC.shape[1]/2)
    xlower = middle - xminhalf
    xupper = middle + xminhalf
    hmapSTIC = hmapSTIC[:,xlower:xupper]
    hmapSTIC = np.array(hmapSTIC)
    np.save('Export/'+identifier + '_HM.npy',hmapSTIC)
    plt.clf()
    cs = plt.contourf(np.flipud(hmapSTIC), cmap='inferno', levels=100)
    cs.axes.set_xlabel('x position [µm]')
    cs.axes.set_ylabel('tau [ms]')
    xBorder = hmapSTIC.shape[1]
    xBorderh = int(xBorder / 2)
    xticks = []
    for xtick in range(-xBorderh, xBorderh, 10):
        xtick = xtick * 50 / 1000  # converting to µm
        xticks.append(xtick)
    xtickpos = range(0, xBorder, 10)
    cs.axes.set_xticks(xtickpos)
    cs.axes.set_xticklabels(xticks)
    # cs.axes.set_xlabel('x position in pixels [a 50 nm]')
    # cs.axes.set_ylabel('tau [ms]')
    # xBorder = hmapSTIC.shape[1]
    # xBorderh = int(xBorder/2)
    # xticks = range(-xBorderh, xBorderh,10)
    # xtickpos = range(0, xBorder, 10)
    # cs.axes.set_xticks(xtickpos)
    # cs.axes.set_xticklabels(xticks)
    plt.ylim(1, )
    plt.yscale('log')
    fig1 = plt.gcf()
    cbar = fig1.colorbar(cs)
    cbar.ax.set_ylabel('STICS function')
    fig1.savefig('Export/'+identifier + '_STICS_HM.jpeg', dpi=300)
    if ShowPlot:
        plt.show()
    #--------------------------------------------------------------------------
    # getting the autocorrelation Curve: take x=0 part of STICS and normalize
    #--------------------------------------------------------------------------
    ACC = STIC[:, 0]
    ACC = ACC[:int(ACC.shape[0] / 2)]
    ACC = ACC - np.amin(ACC)
    ACC = ACC / np.amax(ACC)
    return [ACC, hmapSTICfw]


def Generate_timeline(LS):
    """
    Generates a time-line next to the measured intensities according to the acquisition settings (LineTime)
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
    :param STICS: As nparray
    :param timeline: tau values in millisec
    :return: None
    """
    global max_tau_Gaussian_in_ms
    global LineTime
    global Pixellength
    global identifier
    global ShowPlot

    maxTimelineIndex = int(max_tau_Gaussian_in_ms / LineTime)
    Tau_MSD = []
    Tau_sigma = []
    STICSMSD = np.flipud(STICS)
    x = np.array(range(STICS.shape[1]))
    plt.plot(x, STICSMSD[0, :])
    if ShowPlot:
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
        Tau_sigma.append([tau, sigma])
        if tauindex <= 1000: #for the 3D representation
            if tauindex % 25 == 0:
                BestFitList.append([x, result.best_fit])
                tauaxis.append(tau)
        if tau < 1: # as optical user feedback of the fitting quality
            plt.plot(x, result.best_fit)
            if ShowPlot:
                plt.show()
            plt.clf()
        print(tauindex)
    Tau_sigma_Ex = pd.DataFrame(Tau_sigma, columns=['tau_s', 'sigma'])
    Tau_sigma_Ex.to_csv(path_or_buf='Export/' + identifier + '_Tausigma.csv',
                      header=['tau_s', 'MSD_µm^2'], sep=';', index=False)
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
    if ShowPlot:
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
    Tau_MSD_Ex = pd.DataFrame(Tau_MSD, columns=['tau_s', 'MSD_µm^2','SD'])
    Tau_MSD_Ex.to_csv(path_or_buf='Export/'+identifier + '_TauMSD.csv', header=['tau_s', 'MSD_µm^2', 'SD'], sep=';', index=False)
    if ShowPlot:
        plt.errorbar(Tau_MSD[:, 0], Tau_MSD[:, 1])
        plt.show()
    return


def SumPlotArray_x(LS):
    """
    Sums all Intensity per time values for every x position and plots that
    thus enables the user to check for Vesicles or alike that would disrupt analysis if not excluded.
    Saves plot to PDF.
    :param LS: Unprocessed LS Intensity Matrix
    :return: None
    """
    global identifier
    global ShowPlot

    @jit(nopython=True)
    def Calc_x_yAxis_in_Numba(Pixellenght, LS):
        xAxis = []
        yAxis = []
        for x in range(LS.shape[1]):
            IntensitySum = 0
            for intensity in LS[:, x]:
                IntensitySum = IntensitySum + intensity
            yAxis.append(IntensitySum)
            xAxis.append(x * Pixellength)
        return (xAxis, yAxis)

    Axes = Calc_x_yAxis_in_Numba(Pixellength, LS)
    xAxis = Axes[0]
    yAxis = Axes[1]
    plt.errorbar(xAxis, yAxis)
    plt.title('Intensity-sum over line width')
    plt.xlabel('Line [nm]')
    plt.ylabel('Intensity')
    plt.grid(True)
    fig1 = plt.gcf()
    fig1.savefig('Export/'+identifier + '_Intensity_x_Plot.png', dpi=600)
    if ShowPlot:
        plt.show()
    plt.clf()
    return


def SumPlotArray_t(LS):
    """
    Sums all Intensity per position for every time position and plots that
    thus enables the user to check for bleaching or alike that would disrupt analysis if not excluded.
    Saves plot to PDF.
    :param LS: Unprocessed LS Intensity Matrix
    :return: None
    """
    global LineTime
    global identifier
    global ShowPlot

    @jit(nopython=True)
    def Calc_x_yAxis_in_Numba(Pixellenght, LS):
        xAxis = []
        yAxis = []
        global LineTime
        for y in range(LS.shape[0]):
            IntensitySum = 0
            for intensity in LS[y, :]:
                IntensitySum = IntensitySum + intensity
            yAxis.append(IntensitySum)
            xAxis.append(y * LineTime)
        return (xAxis, yAxis)

    Axes = Calc_x_yAxis_in_Numba(Pixellength, LS)
    xAxis = Axes[0]
    yAxis = Axes[1]
    plt.errorbar(xAxis, yAxis)
    plt.title('Intensity-sum over time')
    plt.xlabel('time [ms]')
    plt.ylabel('Intensity')
    plt.grid(True)
    fig1 = plt.gcf()
    fig1 = plt.gcf()
    #fig1.savefig('Export/' + identifier + '_Intensity_t_Plot.png',dpi=600)
    fig1.savefig('Export/'+identifier + '_Intensity_t_Plot.svg')
    if ShowPlot:
        plt.show()
    plt.clf()
    return


def ACFunction3D(tau, GO, D, waistO, waistZ):
    return GO * (1/(1+((4*D*tau)/(waistO**2)))) * (1/(np.sqrt(1+((4*D*tau)/(waistZ**2)))))


def ACFunction2D(tau, GO, D, waistO, waistZ):
    return GO * (1/(1+((4*D*tau)/(waistO**2))))


def FitACFunctionGetD(STICS,plotter):
    """
    Takes the x=0 , normalizes the data between 0 and 1 and depending on the settings, fits either to 3D Autocorrelation function or to 2D Autocorrelation function. reads out diffusion constant.
    :param STICS: The x=0 ACC
    :param plotter: if it shall be plotted
    :return: D
    """
    global identifier
    global ShowPlot
    STICS = np.array(STICS[:, :])
    taus = STICS[1:, 0]
    #normalizing the data to 0 to 1
    AC = np.array(STICS[1:, 1])
    ACmin = np.min(AC)
    AC = (AC - ACmin)
    ACmax = np.max(AC)
    AC = AC/ACmax
    plt.clf()
    if plotter == 1:
        plt.plot(taus, AC)
    gmodel = lmfit.Model(ACFunction2D)
    params = gmodel.make_params(GO=1, D=1, waistO=330, waistZ=1120) #needs to go to GUI for user setting
    params['waistO'].vary = False
    params['waistZ'].vary = False
    params['GO'].vary = False
    result = gmodel.fit(AC, params, tau=taus)
    params = dict(result.values)
    D = params['D']
    if plotter == 1:
        plt.plot(taus, result.best_fit)
        plt.xscale('log')
        plt.xlabel('time [ms]')
        plt.ylabel('G(tau) normalized')
        text = 'D when calculated by AC over x=0:\n' + str(D/1000) + 'µm^2/s'
        plt.text(1, 0, text, fontsize=10)
        fig1 = plt.gcf()
        fig1.savefig('Export/'+identifier + '_AC_Fit.jpeg', dpi=300)
        if ShowPlot:
            plt.show()
        plt.clf()
    # file = open('AnalyzedData.csv', 'a')
    # file.write('D when calculated by AC over x=0:\n' + str(D/1000) + 'µm^2/s')
    # file.close()
    return D

def AnalysisFunction(LS, starttime, identifier, xstart, xstop, tstart, tstop):
    global HistogramList
    global DValues
    file = open('AnalyzedData.csv', 'a')
    file.write('Analysis File Identifier:' + identifier + '\n')
    file.write('xstop:' + str(xstop) + '\n')
    file.close()
    #Calculate tvalues
    file = open('AnalyzedData.csv', 'a')
    file.write('tstart:' + str(tstart) + '\n')
    file.close()
    if not tstart == 0:
        tstart = int(float(tstart)/ LineTime)
    file = open('AnalyzedData.csv', 'a')
    file.write('tstop:' + str(tstop) + '\n')
    file.close()
    if not tstop == -1:
        tstop = int(float(tstop)/ LineTime)
    LS = LS[tstart:tstop, :]  # excludes depending on user input
    #Intensity Analysis
    IntHist = np.histogram(LS.flatten(), bins=255, range=(0,255), density=False)
    HistogramList.append((identifier,IntHist[0]))
    #--------------------------------------------------------------------------
    # actual STICS calculation
    #--------------------------------------------------------------------------
    ACCSTICS = CalculateSTICS(LS)
    ACC = ACCSTICS[0]
    STICS = ACCSTICS[1] 
    endtime = time.time()
    print(endtime - starttime)
    # -------------------------------------------------------------------------
    # Bins and exports the AC curve as csv for the user
    # -------------------------------------------------------------------------
    ACC = Generate_timeline(ACC)
    ACC = ACC[1:, :]  # remove tau=0 element as this contains the sum of the FFT
    timeline = ACC[:, 0]
    PlotLog = int(np.log10(BinSeconds * 1000))
    Bins = np.logspace(-1, PlotLog, num=100, endpoint=False, base=10)
    StartIndex = 0
    Tau_mean_previous = 0
    BinnedSTICSMean = []
    for Bin in Bins:
        for ACindex, AC in enumerate(timeline):
            if AC >= Bin:
                Endindex = ACindex
                break
        Tau_mean = np.mean(timeline[StartIndex:Endindex + 1])
        G_mean = np.mean(ACC[StartIndex:Endindex + 1, 1])
        G_SD = np.std(ACC[StartIndex:Endindex + 1, 1])
        if not Tau_mean_previous == Tau_mean:
            BinnedSTICSMean.append([Tau_mean, G_mean, G_SD])
        Tau_mean_previous = Tau_mean
        StartIndex = Endindex
    BinnedSTICSMean = np.array(BinnedSTICSMean)
    xNullEx = pd.DataFrame(BinnedSTICSMean,
                           columns=['tau', 'G_tau_Average', 'SD'])
    xNullEx.to_csv(path_or_buf='Export/'+identifier + '_xNullEx.csv',
                   header=['tau', 'AC_Average', 'SD'], sep=';', index=False)
    ACC=BinnedSTICSMean[:, :2]
    # -------------------------------------------------------------------------
    # Fitting of the Diffusion models over the AC curve for D extraction
    # -------------------------------------------------------------------------
    AreatoFitms = BinSeconds*1000 # to only look at a certain area for fitting: cut the binned array accordingly
    absolute_val_array = np.abs(ACC[:, :1] - AreatoFitms)
    smallest_difference_index = absolute_val_array.argmin()
    STICforD = ACC[:smallest_difference_index, :]
    D = FitACFunctionGetD(STICforD, 1)
    print(D, 'nm^2/ms')
    D = D/1000
    print(D, 'µm^2/s')
    DValues.append((identifier,D))
    #--------------------------------------------------------------------------
    # Gaussian fitting and MSD extraction
    #--------------------------------------------------------------------------
    GaussianFit(STICS, timeline)

def main(args):
    global Hz_Aquisition
    global LineTime
    global BinSeconds
    global Pixellength
    global max_tau_Gaussian_in_ms
    global identifier
    global HistogramList
    global ShowPlot
    global DValues
    #--------------------------------------------------------------------------
    #those settings should be included in the GUI and need to be checked by the user
    #--------------------------------------------------------------------------
    max_tau_Gaussian_in_ms = 1000
    BinSeconds = 1
    Hz_Aquisition = 1800
    LineTime = 1000 / Hz_Aquisition  # in millisec
    Pixellength = 50  # nm
    #--------------------------------------------------------------------------
    # start of the analysis
    #--------------------------------------------------------------------------
    starttime = time.time()
    print(starttime)
    LSOrig = np.array(readFile()) # File imported to Matrix
    if not args.auto: #manual analysis
        LS = LSOrig
        ShowPlot = True
        identifier = input('Give an identifier for flagging the data exports: ')
        SumPlotArray_x(LS) # for excluding inhomogeneities
        xstart = int(input('Give starting point [nm]')) or 0
        xstop = int(input('Give stopping point [nm]')) or -1
        if not xstop == -1:
            xstop = int(xstop / Pixellength)
        if not xstart == 0:
            xstart = int(xstart / Pixellength)
        LS = LS[:,xstart:xstop] # excludes Vesicels etc depending on user input
    #Now let the user check which time interval shall be considered (because of bleaching etc)
        SumPlotArray_t(LS)
        tstart = input('Give starting time [ms]') or 0
        tstop = input('Give stopping time [ms]') or -1
        AnalysisFunction(LS,starttime,identifier,xstart,xstop,tstart,tstop)
    else:
        #AUTOMATIC ANALYSIS
        argInput = pd.read_excel('./data/Analysis_Args.xlsx', sheet_name='Tabelle1',header=0)
        for index, row in argInput.iterrows():
            LS = LSOrig
            identifier = row['Name']
            xstart = row['xstart']
            xstop = row['xstop']
            tstart = row['tstart']
            tstop = row['tstop']
            xstart = int(xstart / Pixellength)
            xstop = int(xstop / Pixellength)
            LS = LS[:,xstart:xstop] # excludes Vesicels etc depending on user input
    #Now let the user check which time interval shall be considered (because of bleaching etc)
            AnalysisFunction(LS,starttime,identifier,xstart,xstop,tstart,tstop)
    file = open('AnalyzedData.csv', 'a') # logging user settings for reproducibility
    file.write('xstart:' + str(xstart) + '\n')
    file.close()
    if not xstart == 0:
        xstart = int(xstart)
        xstart = int(xstart / Pixellength)
    filename = identifier.split('_TP')[0]
    np.savetxt(f'Export/{filename}_ACFit', DValues, delimiter=",", fmt='%s')
    data_dict = {identifier: values for identifier, values in HistogramList}
    pd.DataFrame(data_dict).to_csv(f'Export/{filename}_Pixelhistogram.csv')
    #Calculate tstart
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--auto', help='Start automatic parsing', action='store_true')
    args = parser.parse_args()
    main(args)
