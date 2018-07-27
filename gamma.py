'''
Module with functions for analysis of gamma flood data. If running this 
file as a script, it will run a complete data analysis pipeline on the
supplied gamma flood data.
'''

# Packages for making life easier
import os.path
import string

# Data analysis packages
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting

# Plotting packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


sns.set_context('talk')
sns.set_style("ticks")
sns.set_palette("colorblind")

class Line:
    '''
    A class for spectral lines. The infomation in each instance will help
    supply parameters for fitting peaks.

    Attributes:
        source: str
            The name of the radioactive source producing the line. 
            E.g., 'Am241'
        energy: float
            The energy of the line in keV.
        chan_low: int
            When searching the channel spectrum for peaks, channels below
            'chan_low' will be ignored.
        chan_high: int
            When searching the channel spectrum for peaks, channels above
            'chan_high' will be ignored.
    '''
    def __init__(self, source, energy, chan_low, chan_high, latex):
        self.source = source
        self.energy = energy
        self.chan_low = chan_low
        self.chan_high = chan_high
        self.latex = latex

# Defining 'Line' instances for Am241 and Co57. 
am = Line('Am241', 59.54, chan_low=3000, chan_high=6000,
    latex=r'${}^{241}{\rm Am}$')
# Co57's 'chan_low' and 'chan_high' attributes have not been tested.
co = Line('Co57', 122.06, chan_low=5000, chan_high=8000,
    latex=r'${}^{57}{\rm Co}$')

# TODO 
# Make a class that contains information about plot titles?
# Also maybe make path construction object oriented. These things might
# drastically reduce the number of parameters needed.


def construct_path(filepath,  description='', etc='', ext='', save_dir=''):
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

    Keyword Arguments:
        ext: str
            The file name extension.
        description: str
            A short description of what the file contains. This will be 
            prepended to the file name.
            (default: '')
        etc: str 
            Other important information, e.g., pixel coordinates. This will 
            be appended to the file name.
            (default: '')
        save_dir: str
            The directory to which the file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')

    Return:
        save_path: str
            A Unix/Linux/MacOS style path that can be used to save data
            and plots in an organized way.
    '''
    ### Handling exceptions and potential errors

    # If 'ext' does not start with a '.', fix it.
    if ext and ext[0] != '.':
        ext = f'.{ext}'

    # Check that the save directory exists
    if save_dir:
        if not os.path.exists(save_dir):
            raise ValueError(f'The directory \'{save_dir}\' does not exist.')

    ### Constructing the path name

    # Construct the file name from the file name in 'filepath'.
    filename = os.path.basename(filepath)
    save_path = os.path.splitext(filename)[0]

    # Map all whitespace characters and '.' to underscores
    trans = str.maketrans(
        '.' + string.whitespace, 
        '_' * (len(string.whitespace) + 1)
    )
    save_path = save_path.translate(trans)
    
    # Prepend the description if specified
    if description:
        save_path = f'{description}_{save_path}'

    # Append extra info to the file name if specified
    if etc:
        save_path += f'_{etc}'


    # Append the file extension
    save_path += ext

    # Prepend the save directory if specified
    if save_dir:
        save_path = f'{save_dir}/{save_path}'

    return save_path


def count_map(filepath, save=True, path_constructor=construct_path,
    etc='', ext='.txt', save_dir=''):
    '''
    Generates event count data for each pixel for raw gamma flood data.

    Arguments:
        filepath: str
            The filepath to the gamma flood data.

    Keyword Arguments:
        save: bool 
            If True, saves count_map as an ascii file.
            (default: True)
        path_constructor: function
            A function that takes the same parameters as the function
            'gamma.construct_path' that returns a string representing a 
            path to which the file will be saved.
            (default: gamma.construct_path)
        etc: str 
            Other important information. Will be appended to the file name.
            (default: '')
        save_dir: str
            The directory to which the count_map file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        ext: str
            The file name extension for the count_map file. 
            (default: '.txt')

    Return:
        count_map: 2D numpy.ndarray
            A 32 x 32 array of floats. Each entry represents the number of
            counts read by the detector pixel at the corresponding index.
    '''
    # Generating the save path, if needed.
    if save:
        save_path = path_constructor(filepath, ext=ext,
            save_dir=save_dir, etc=etc, description='count_data')

    # Get data from gamma flood FITS file
    with fits.open(filepath) as file:
        data = file[1].data

    # The masks ('mask', 'PHmask', 'STIMmask', and 'TOTmask') below show 
    # 'True' or '1' for valid/desired data points, and 'False' or '0' 
    # otherwise. Note that this is the reverse of numpy's convention for 
    # masking arrays.

    # 'START' and 'END' denote the indices between which 'data['TEMP']'
    # takes on a resonable value. START is the first index with a 
    # temperature greater than -20 C, and END is the last such index.
    mask = data['TEMP'] > -20
    START = np.argmax(mask)
    END = len(mask) - np.argmax(mask[::-1])

    # Masking out non-positive pulse heights
    PHmask = 0 < np.array(data['PH'][START:END])
    # Masking out artificially stimulated events
    STIMmask = np.array(data['STIM'][START:END]) == 0
    # Combining the above masks
    TOTmask = np.multiply(PHmask, STIMmask)

    # Generate the count_map from event data
    count_map = np.zeros((32, 32))

    for i in range(32):
        RAWXmask = np.array(data['RAWX'][START:END]) == i
        for j in range(32):
            RAWYmask = np.array(data['RAWY'][START:END]) == j
            count_map[i, j] = np.sum(np.multiply(
                TOTmask, np.multiply(RAWXmask, RAWYmask)))

    # Saves the 'count_map' array as an ascii file.
    if save:
        np.savetxt(save_path, count_map)

    return count_map


def quick_gain(filepath, line, path_constructor=construct_path, 
    save_plot=True, plot_dir='', plot_ext='.eps', 
    save_data=True, data_dir='', data_ext='.txt',
    etc=''):
    '''
    Generates gain correction data from the raw gamma flood event data.
    Currently, the fitting done might fail for sources other than Am241.

    Arguments:
        filepath: str
            The filepath to the gamma flood data
        line: an instance of Line
            The attributes of 'line' will provide information for fitting.

    Keyword Arguments:
        save_plot: bool
            If true, plots and energy spectrum for each pixel and saves
            the figure.
            (default: True)
        plot_dir: str
            The directory to which the plot file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        plot_ext: str
            The file name extension for the plot file.
            (default: '.eps')  
        save_data: bool 
            If True, saves gain data as a .txt file.
            (default: True)
        data_dir: str
            The directory to which the gain file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        data_ext: str
            The file name extension for the gain file. 
            (default: '.txt')
        etc: str 
            Other important information
            (default: '')

    Return:
        gain: 2D numpy.ndarray
            A 32 x 32 array of floats. Each entry represents its respective 
            pixel's gain, where channels * gain = energy
    '''

    if save_data:
        data_path = path_constructor(filepath=filepath, ext=data_ext,
            description='gain_data', etc=etc)

    if save_plot:
        plot_path = path_constructor(filepath=filepath, description='gain',
            etc=etc)


    # Get data from gamma flood FITS file
    with fits.open(filepath) as file:
        data = file[1].data

    mask = data['TEMP'] > -20

    START = np.argmax(mask)
    END = len(mask) - np.argmax(mask[::-1])

    maxchannel = 10000
    bins = np.arange(1, maxchannel)
    gain = np.zeros((32, 32))

    # Iterating through pixels
    for x in range(32):
        RAWXmask = data.field('RAWX')[START:END] == x
        for y in range(32):
            # Getting peak height in 'channels' for all events for the 
            # current pixel.
            channel = data.field('PH')[START:END][np.nonzero(
                np.multiply(
                    RAWXmask, data.field('RAWY')[START:END] == y
                ))]

            # If there were events at this pixel, fit the strongest peak
            # in the channel spectrum with a Gaussian.
            if len(channel):
                # 'spectrum' contains counts at each channel
                spectrum, edges = np.histogram(channel, bins=bins, 
                    range=(0, maxchannel))
                # 'centroid' is the channel with the most counts in the 
                # interval between 'line.chan_low' and 'line.chan_high'.
                centroid = np.argmax(spectrum[line.chan_low:line.chan_high])\
                    + line.chan_low
                fit_channels = np.arange(centroid - 100, centroid + 200)
                g_init = models.Gaussian1D(amplitude=spectrum[centroid], 
                    mean=centroid, stddev=75)
                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, fit_channels, spectrum[fit_channels])

                # If we can determine the covariance matrix (which implies
                # that the fit succeeded), then calculate this pixel's gain
                if fit_g.fit_info['param_cov'] is not None:
                    gain[y, x] = line.energy / g.mean
                    # Plot each pixel's spectrum
                    if save_plot:
                        plt.figure()
                        sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
                        fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
                        mean_err = np.diag(fit_g.fit_info['param_cov'])[1]
                        frac_err = np.sqrt(np.square(fwhm_err) 
                            + np.square(g.fwhm * mean_err / g.mean)) / g.mean
                        str_err = str(int(round(frac_err * line.energy * 1000)))
                        str_fwhm = str(int(round(
                                line.energy * 1000 * g.fwhm / g.mean, 0
                        )))
                        plt.text(
                            maxchannel * 3 / 5, spectrum[centroid] * 3 / 5, 
                            r'$\mathrm{FWHM}=$' + str_fwhm + r'$\pm$' 
                            + str_err + ' eV', fontsize=13
                        )

                        plt.hist(
                            np.multiply(channel, gain[y, x]), 
                            bins=np.multiply(bins, gain[y, x]),
                            range=(0, maxchannel * gain[y, x]), 
                            histtype='stepfilled'
                        )

                        plt.plot(
                            np.multiply(fit_channels, gain[y, x]), 
                            g(fit_channels), label='Gaussian fit'
                        )

                        plt.ylabel('Counts')
                        plt.xlabel('Energy')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{plot_path}_x{x}_y{y}{plot_ext}')
                        plt.close()

    # Interpolate gain for pixels where fit was unsuccessful. Do it twice in
    # case the first pass had insufficient data to interpolate some pixels.
    for _ in range(2):
        newgain = np.zeros((34, 34))
        # Note that newgain's indices will be shifted over one from 'gain'.
        newgain[1:33, 1:33] = gain
        # 'empty' contains indices at which the fit was unsuccessful
        empty = np.transpose(np.nonzero(gain == 0.0))
        # Iterating through pixels with failed fitting.
        for x in empty:
            # 'temp' is the 3x3 array of gain values around the pixel for 
            # which the fitting failed.
            temp = newgain[x[0]:x[0]+3, x[1]:x[1]+3]
            # If there are any nonzero values in 'temp', set the pixel's 
            # gain to their mean.
            if np.count_nonzero(temp):
                gain[x[0], x[1]] = np.sum(temp) / np.count_nonzero(temp)

    # Save gain data to an ascii file.
    if save_data:
        np.savetxt(data_path, gain)

    return gain


def get_spectrum(filepath, gain, bins=10000, energy_range=(0.01, 120), 
    path_constructor=construct_path, save=True, ext='.txt', save_dir='',
    etc=''):
    '''
    Applies gain correction to get energy data, and then bins the events
    by energy to obtain a spectrum.

    Arguments:
        filepath: str
            The filepath to the gamma flood data
        gain: 2D numpy.ndarray
            A 32 x 32 array of floats. Each entry represents its respective 
            pixel's gain, where channels * gain = energy

    Keyword Arguments:
        bins: int
            Number of energy bins
        energy_range: tuple of numbers
            The bins will be made between these energies
        path_constructor: function
            A function that takes the same parameters as the function
            'gamma.construct_path' that returns a string representing a 
            path to which the file will be saved.
            (default: gamma.construct_path)
        save:
            If True, 'spectrum' will be saved as an ascii file. Parameters 
            relevant to this saving are below
        etc: str 
            Other important information to go in the filename.
            (default: '')
        save_dir: str
            The directory to which the  file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        ext: str
            The file name extension for the count_map file. 
            (default: '.txt')

    Return:
        spectrum: 2D numpy.ndarray
            This array represents a histogram wrt the energy of an event.
            spectrum[0] is a 1D array of counts in each bin, and spectrum[1] 
            is a 1D array of the middle enegies of each bin in keV. E.g., if 
            the ith bin counted events between 2 keV and 4 keV, then the 
            value of spectrum[1, i] is 3.
    '''
    # Generating the save path, if needed.
    if save:
        save_path = path_constructor(ext=ext, filepath=filepath, 
            save_dir=save_dir, etc=etc, description='spectrum')

    # Adding a buffer of zeros around the 'gain' array. (Note that the
    # indices will now be shifted over by one.)
    gain_buffed = np.zeros((34, 34))
    gain_buffed[1:33, 1:33] = gain
    gain = gain_buffed

    # Get data from gamma flood FITS file
    with fits.open(filepath) as file:
        data = file[1].data

    # 'START' and 'END' denote the indices between which 'data['TEMP']'
    # takes on a resonable value. START is the first index with a 
    # temperature greater than -20 C, and END is the last such index.
    temp_mask = data['TEMP'] > -20
    START = np.argmax(temp_mask)
    END = len(temp_mask) - np.argmax(temp_mask[::-1])

    # If there's gain data then correct the spectrum
    # PH_COM is a list of length 9 corresponding to the charge in pixels 
    # surrounding the event.
    #
    # PH_COM -> gain correct -> sum positive elements in the 3x3 array -> event in energy units
    energies = []
    for event in data[START:END]:
        row = event['RAWY']
        col = event['RAWX']
        temp = event['PH_COM'].reshape(3, 3)
        mask = (temp > 0).astype(int)
        energies.append(np.sum(
            np.multiply(
                np.multiply(mask, temp), 
                gain[row:row + 3, col:col + 3])
            )
        )

    # Binning by energy
    counts, edges = np.histogram(energies, bins=bins, range=energy_range)

    # Getting the midpoint of the edges of each bin.
    midpoints = (edges[:-1] + edges[1:]) / 2

    # Consolidating 'counts' and 'midpoints' into a 2D array 'spectrum'.
    spectrum = np.empty((2, counts.size))
    spectrum[0, :] = counts
    spectrum[1, :] = midpoints

    if save:
        np.savetxt(save_path, spectrum)

    return spectrum


def plot_spectrum(spectrum, line, title='Spectrum', save=True, 
    path_constructor=construct_path, filepath='', etc='', ext='.eps', 
    save_dir=''):
    '''
    Fits and plots the spectrum returned from 'get_spectrum'. To show the 
    plot with an interactive interface, call 'plt.show()' right after 
    calling this function.

    Arguments:
        spectrum: 2D numpy.ndarray
            This array represents a histogram wrt the energy of an event.
            spectrum[0] is a 1D array of counts in each bin, and spectrum[1] 
            is a 1D array of the middle enegies of each bin in keV. E.g., if 
            the ith bin counted events between 2 keV and 4 keV, then the 
            value of spectrum[1, i] is 3.
        line: an instance of Line
            The attributes of 'line' will provide information for fitting.

    Keyword Arguments:
        save:
            If True, 'spectrum' will be saved as an ascii file. Parameters 
            relevant to this saving are below
        path_constructor: function
            A function that takes the same parameters as the function
            'gamma.construct_path' that returns a string representing a 
            path to which the file will be saved.
            (default: gamma.construct_path)
        filepath: str
            This string will form the basis for the file name in the path 
            returned by this function. If a path is supplied here, the 
            file name sans extension will be trimmed out and used.
        etc: str 
            Other important information to go in the filename.
            (default: '')
        save_dir: str
            The directory to which the  file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        ext: str
            The file name extension for the count_map file. 
            (default: '.eps')

    '''
    # Constructing a save path, if needed
    if save:
        save_path = path_constructor(ext=ext, filepath=filepath, 
            save_dir=save_dir, etc=etc, description='energy_spectrum')

    maxchannel = 10000

    # 'centroid' is the bin with the most counts
    centroid = np.argmax(spectrum[0, 1000:]) + 1000
    # Fit in an asymetrical domain about the centroid to avoid 
    # low energy tails.
    fit_channels = np.arange(centroid - 80, centroid + 150)
    # Do the actual fitting.
    g_init = models.Gaussian1D(amplitude=spectrum[0, centroid], 
        mean=centroid, stddev=75)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, fit_channels, spectrum[0, fit_channels])

    sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
    fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
    mean_err = np.diag(fit_g.fit_info['param_cov'])[1]
    frac_err = np.sqrt(
        np.square(fwhm_err) 
        + np.square(g.fwhm * mean_err / g.mean)
    ) / g.mean

    # Displaying the FWHM on the spectrum plot, with error.
    display_fwhm = str(int(round(line.energy * 1000 * g.fwhm / g.mean, 0)))
    display_err  = str(int(round(frac_err * line.energy * 1000)))

    plt.text(70, spectrum[0, centroid] * 3 / 5, r'$\mathrm{FWHM}=$' 
        +  display_fwhm + r'$\pm$' + display_err + ' eV', 
        fontsize=13)

    plt.plot(spectrum[1], spectrum[0], label=line.latex)
    plt.plot(spectrum[1, fit_channels], g(fit_channels), 
        label = 'Gaussian fit')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.legend()

    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.close()


def pixel_map(values, value_label, title='', save=True, filepath='', 
    path_constructor=construct_path, etc='', ext='.eps', save_dir=''):
    '''
    Construct a heatmap of counts across the detector using matplotlib.

    Arguments:
        values: 2D array
            A 32 x 32 array of numbers.
        value_label: str
            A short label denoting what data is supplied in 'values'.
            In the hover tooltip, the pixel's value will be labelled with 
            this string. E.g., if value_label = 'Counts', the tooltip might
            read 'Counts: 12744'. The color bar title is also set to this 
            value. Also, value_label.lower() is prepended to the file name
            if saving the plot.

    Keyword Arguments:
        title: str
            The title displayed on the plot.
            (default: 'Pixel Map')
        save: bool
            If True, saves the Bokeh plot as an HTML file.
        filepath: str
            This string will form the basis for the file name in the path 
            returned by this function. If a path is supplied here, the 
            file name sans extension will be trimmed out and used.
        path_constructor: function
            A function that takes the same parameters as the function
            'gamma.construct_path' that returns a string representing a 
            path to which the file will be saved.
            (default: gamma.construct_path)
        etc: str 
            Other important information. Will be appended to the file name.
            (default: '')
        save_dir: str
            The directory to which the count_map file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
    '''
    # Generate a save path, if needed.
    if save:
        description = (value_label.lower() + '_map').replace(' ', '_')
        save_path = path_constructor(filepath, ext=ext, 
            description=description, save_dir=save_dir, etc=etc)

    plt.figure()
    masked = np.ma.masked_values(values, 0.0)
    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='gray')
    plt.imshow(masked)
    c = plt.colorbar()
    c.set_label(value_label)
    plt.title(title)
    plt.tight_layout()

    if save:
        plt.savefig(save_path)


def count_hist(count_map, bins=100, title='Count Histogram', save=True, 
    filepath='', etc='', ext='.eps', path_constructor=construct_path, 
    save_dir=''):

    # Generate a save path, if needed.
    if save:
        save_path = path_constructor(filepath, ext=ext, etc=etc,
            description='count_hist', save_dir=save_dir)

    plt.figure()
    plt.hist(np.array(count_map).flatten(), bins=bins, 
        range=(0, np.max(count_map) + 1), 
        histtype='stepfilled')
    plt.ylabel('Pixels')
    plt.xlabel('Counts')
    plt.title(title)
    plt.tight_layout()

    if save:
        plt.savefig(save_path)


# If this file is run as a script, the code below will a complete pipeline
# for gamma flood data analysis with default parameter values.
if __name__ == '__main__':

    filepath = input('Enter the filepath to the gamma flood data: ')
    source = input('Enter the name of the source used (Am241 or Co57): ')

    # Getting the 'Line' instance that corresponds to 'source'.
    if source.lower() == am.source.lower():
        line = am
    elif source.lower() == co.source.lower():
        line = co
    else:
        raise ValueError('''
            This module doesn't explicitly support that source yet. To 
            specify a custom source, instantiate a 'Line' object and pass
            it to the 'quick_gain' and 'plot_spectrum' functions where 
            indicated.
        ''')
        
    pixel_dir = input('Enter a directory to save pixel spectra to: ')

    # Processing data
    print('Calculating count data...')
    count_map = count_map(filepath)
    print('Calculating gain data...')
    gain = quick_gain(filepath, line, save_dir=pixel_dir)
    print('Calculating the energy spectrum...')
    spectrum = get_spectrum(filepath, gain)

    # Plotting
    print('Plotting...')
    plot_spectrum(spectrum, line, filepath=filepath)

    count_hist(count_map, filepath=filepath)

    pixel_map(count_map, 'Counts', filepath=filepath)
    
    pixel_map(gain, 'Gain', filepath=filepath)

    print('Done!')