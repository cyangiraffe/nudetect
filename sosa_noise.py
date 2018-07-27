from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.modeling import models, fitting
import os.path
import pickle
import seaborn as sns
from matplotlib.gridspec import GridSpec


# need to check if detector exists
#automatically make the directories

def noise(filepath, gainpath, output_path,pos, detector="",sep_by_detector=False, etc=""):
    '''
    Constructs a path for saving data and figures based on user input. The
    main use of this function is for other functions in this package to use
    it as the default path constructor if the user doesn't supply their own
    function.

    Note to developers: This function is designed to throw a lot of
    exceptions and be strict about formatting early on to avoid
    complications later. Call it early in scripts to avoid losing the
    results of a long computation to a mistyped directory.

    Arguments:
        filepath: str
            This string will form the basis for the file name in the path
            returned by this function. If a path is supplied here, the
            file name sans extension will be trimmed out and used.

        gainapth: str
            Path to the gain data.

        output_path: str
            Working directory where all subdirectories and data created will be stored.
        detector: str
            The detector ID. Required if sep_by_detector == True.
        pos: int
            detector position?


    Keyword Arguments:
        ext: str
            The file name extension.
        description: str
            A short description of what the file contains. This will be
            prepended to the file name.
            (default: '')
        etc: str
            Other important information, e.g., pixel coordinates, voltage This will
            be appended to the file name.
            (default: '')
        sep_by_detector: bool
            If True, constructs the file path such that the file is saved in
            a subdirectory of 'save_dir' named according to the string
            passed for 'detector'. Setting this to 'True' makes 'detector' a
            required kwarg.
            (default: False)
        detector: str
            The detector ID.

    Return:
        save_path: str
            A Unix/Linux/MacOS style path that can be used to save data
            and plots in an organized way.
    '''

    #Set the seaborn framework and clean some parameters for use.
    sns.set_context('talk')
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    filepath = filepath.strip()
    gainpath = gainpath.strip()
    if output_path[-1] != "/": output_path.append("/")
    if type(pos) is not int:
        pos = int(pos.strip())
    #Check if file exists,

    while not os.path.exists(filepath):
        filepath = input('Please enter the filepath to the noise data: ').strip()

    #find slash
    slash = 0
    i = 0
    for char in filepath:
        if char == '/':
            slash = i
        i += 1
    #remove whitespace from entry
    #if we want to seperate by detector, pad the detector string with "/"
    if sep_by_detector:

        detector = detector.strip("/").strip()


    #What does gainBool do?
    gainBool = os.path.exists(gainpath)
    gain = np.ones((32,32))

    if gainBool:
        gain = pickle.load(open(gainpath, 'rb'))


    file = fits.open(filepath)
    data = file[1].data
    file.close()



    #Outputs masked array with True values @ indexes w/position=pos
    # and temp > -20
    mask = np.multiply((data['DET_ID'] == pos), (data['TEMP'] > -20))

    START = np.argmax(mask)
    END = len(mask) - np.argmax(mask[::-1])

   #Bins spanning twice the maxchannel value, w/ a 33x33 channel map
    maxchannel = 1000
    bins = np.arange(0-maxchannel,maxchannel)
    #channelMap makes a 33x33 matrix of lists
    channelMap = [ [[] for i in range(33)] for j in range(33) ]

    # start at the first valid index, end at the last valid Index
    # and if UP at that index is valid, then... WHAT IS THIS PARRT?
    for i in np.arange(START, END):
        if data['UP'][i]:
    		for j in range(9):
			channelMap[data['RAWY'][i] + (j//3) - 1][data['RAWX'][i] + (j%3) - 1].append(data['PH_RAW'][i][j])

        '''		channelMap[data['RAWX'][i]-1][data['RAWY'][i]-1].append(data['PH_RAW'][i][0])
		channelMap[data['RAWX'][i]+0][data['RAWY'][i]-1].append(data['PH_RAW'][i][1])
		channelMap[data['RAWX'][i]+1][data['RAWY'][i]-1].append(data['PH_RAW'][i][2])
		channelMap[data['RAWX'][i]-1][data['RAWY'][i]+0].append(data['PH_RAW'][i][3])
		channelMap[data['RAWX'][i]+0][data['RAWY'][i]+0].append(data['PH_RAW'][i][4])
		channelMap[data['RAWX'][i]+1][data['RAWY'][i]+0].append(data['PH_RAW'][i][5])
		channelMap[data['RAWX'][i]-1][data['RAWY'][i]+1].append(data['PH_RAW'][i][6])
		channelMap[data['RAWX'][i]+0][data['RAWY'][i]+1].append(data['PH_RAW'][i][7])
		channelMap[data['RAWX'][i]+1][data['RAWY'][i]+1].append(data['PH_RAW'][i][8])'''


    countMap = [[len(channelMap[j][i]) for i in range(32)] for j in range(32)]
    plt.figure()
    plt.imshow(countMap)
    c = plt.colorbar()
    c.set_label('Counts')

    #FULL WIDTH HALF MAXIMUM MAP!  MAKE DIRECTORY!

    os.makedirs(output_path +  detector + '/pixels/')

    FWHM = []
    FWHM_map = np.array([[np.nan for i in range(32)] for j in range(32)])
    for row in range(32):
        for col in range(32):
            if (channelMap[row][col]):

                #Computes the histogram of each array in the matrix, using maxchannel range bins
                # Whats in the ChannelMap?
                tempSpec = np.histogram(channelMap[row][col], bins=bins, range = (0-maxchannel,maxchannel))
                #Fit a gaussianto the data
                fit_channels = tempSpec[1][:-1]
                g_init = models.Gaussian1D(amplitude=np.max(tempSpec[0]), mean=0, stddev = 75)
                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, fit_channels, tempSpec[0])

                #append the gain-multiplied gaussian full-width half-maxmum (FWHM) to a list.
                FWHM.append(g.fwhm * gain[row][col])

                #only have values in the FWHM map that are less than 1000.
                if g.fwhm < 1000:
                    FWHM_map[row][col] = g.fwhm * gain[row][col]
                plt.hist(np.multiply(channelMap[row][col], gain[row][col]), bins = np.multiply(bins, gain[row][col]), range = (0-maxchannel*gain[row][col],maxchannel*gain[row][col]), histtype='stepfilled')
                plt.plot(np.multiply(fit_channels, gain[row][col]), g(fit_channels))
                plt.ylabel('Counts')
                if gainBool:
                    plt.xlabel('Energy (keV)')
                    plt.tight_layout()
                    plt.savefig(output_path + detector + '/pixels/' + filename[:-4] + 'x' + str(col) + 'y' + str(row) + '_spec_gain.eps')
                else:
                    plt.xlabel('Channel')
                    plt.tight_layout()
                    plt.savefig( output_path + detector + '/pixels/' + filename[:-4] + 'x' + str(col) + 'y' + str(row) + '_spec_corr.eps')
                plt.close()

    plt.figure()
    #if valid gainpath provided, plot a histogram of energies, otherwise use channels.
    if gainBool:
        plt.hist(FWHM, bins = 50, range = (0, 4), histtype='stepfilled')
        bot, top = plt.ylim()
        left, right = plt.xlim()
        plt.text(right*0.5, top*0.8, 'Mean = ' + str(int(round(np.mean(FWHM) * 1000, 0))) + ' eV', fontsize = 16)
        plt.text(right*0.5, top*0.6, '1-Sigma = ' + str(int(round(np.std(FWHM) * 1000, 0))) + ' eV', fontsize = 16)
        plt.xlabel('FWHM (keV)')

    else:
        plt.hist(FWHM, bins = 50, range = (0, 300), histtype='stepfilled')
        bot, top = plt.ylim()
        left, right = plt.xlim()
        plt.text(right*0.5, top*0.8, 'Mean = ' + str(round(np.mean(FWHM), 0)) + ' channels', fontsize = 16)
        plt.text(right*0.5, top*0.6, '1-Sigma = ' + str(round(np.std(FWHM), 0)) + ' channels', fontsize = 16)
        plt.xlabel('FWHM (channels)')

    plt.ylabel('Pixels')

    plt.tight_layout()


    #Save the FWHM figure
    os.makedirs(output_path + "/det_figs/"+  detector+ "/")

    if gainBool:
        plt.savefig(output_path + "det_figs/"+  detector + filename[:-4] + 'FWHMhist_gain.eps')
    else:
	    plt.savefig(output_path + "det_figs/"+  detector + filename[:-4] + 'FWHMhist_corr.eps')
    #plt.show()
    plt.close()

    os.makedirs(output_path + "detectorData/" + detector)
    outfile = open(output_path + "detectorData/" + detector + '/noise2.out', 'w')
    outfile.write('Mean FWHM: '  +'\n')
    outfile.write(str(np.mean(FWHM)) + '\n')
    outfile.write('Std dev FWHM: '  +'\n')
    outfile.write(str(np.std(FWHM)) + '\n')
    outfile.close()


    #makes sure that this exists
    dumpfile = open(output_path+ "detectorData/" + detector + '/noise2_FWHMmap_dump.txt', 'wb')
    #Numpy.saveTXT
    pickle.dump(FWHM_map, dumpfile)
    dumpfile.close()

    #make a FWHM colormap, masking values less than 5.
    plt.figure()
    masked = np.ma.masked_where(FWHM_map > 5, FWHM_map)
    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='gray')
    plt.imshow(masked)
    c = plt.colorbar()

    #Set FWHM keV metric if there is a gain path, otherwise we use channels as our metric.
    if gainBool:
    	c.set_label('FWHM (keV)')
    else:
	    c.set_label('FWHM (channels)')
    plt.tight_layout()

    #MAuthomatically make a path to save the detector figures in.
    os.makedirs(output_path +"det_figs/"+ detector)
    if gainBool:
	    plt.savefig(output_path +  "det_figs/"+  detector + filename[:-4] + 'FWHMmap_gain.eps')
    else:
        plt.savefig(output_path + "det_figs/"+  detector+ filename[:-4] + 'FWHMmap_corr.eps')

    plt.close()