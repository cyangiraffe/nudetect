'''
Module with functions for analysis of gamma flood data.
PH: Pulse height
STIM: Mask value. If 1, event is from voltage pulse, not photon. (Usually applied to pixel (10, 10))
UP: micro pluse -- also a mask (in noise)
'''

# Packages for making life easier
import os.path

# Data analysis packages
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting

# Bokeh for interactive plots (v. 0.13.0 at time of writing)
import bokeh.io
import bokeh.plotting
import bokeh.models as bm

def construct_path(ext='', filepath='', description='', detector='', 
    source='', temp='', voltage='', save_dir='', etc='', 
    sep_by_detector=False):
    '''
    Constructs a path for saving data and figures based on user input. The 
    main use of this function is for other functions in this package to use 
    it as the default path constructor if the user doesn't supply their own 
    function.

    Note: you must supply at least one of these collections of parameters:
        * filepath
        * detector, source, temp, voltage
    If both are given, only 'filepath' will be considered

    Note to developers: This function is designed to throw a lot of 
    exceptions and be strict about formatting early on to avoid 
    complications later. Call it early in scripts to avoid losing the 
    results of a long computation to a mistyped directory.

    Keyword Arguments:
        ext: str
            The file name extension.
        filepath: str
            A Unix/Mac style path to a file whose name will form the 
            basis for the file name returned by this function, if this
            parameter is specified.
        detector: str
            The detector ID
            (default: '')
        source: str
            The radioactive source used
            (default: '')
        temp: str
            The temperpature
            (default: '')
        voltage: str
            The voltage
            (default: '')
        description: str
            A short description of what the file contains. This will be 
            prepended to the file name.
            E.g., 'count_map'.
            (default: '')
        etc: str 
            Other important information, e.g., pixel coordinates. This will 
            be appended to the file name.
            (default: '')
        save_dir: str
            The directory to which the count_map file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        sep_by_detector: bool
            If True, constructs the file path such that the file is saved in 
            a subdirectory of 'save_dir' named according to the string 
            passed for 'detector'. Setting this to 'True' makes 'detector' a 
            required kwarg, even if 'filepath' is specified.
            (default: False)
    '''
    ### Handling exceptions and potential errors

    # If 'ext' does not start with a '.', fix it.
    if ext and ext[0] != '.'
        ext = f'.{ext}'

    # Check that the save directory exists
    if save_dir and sep_by_detector:
        if not detector:
            raise Exception('''
                Since 'sep_by_detector' is True, a value must be supplied
                for the 'detector' parameter.
            ''')
        if not os.path.exists(f'{save_dir}/{detector}'):
            raise ValueError(
                f'The directory \'{save_dir}/{detector}\' does not exist.'
            )
    elif save_dir:
        if not os.path.exists(save_dir):
            raise ValueError(f'The directory \'{save_dir}\' does not exist.')

    # Raise an exception if not enough parameters are supplied.
    if not (filepath or (detector and source and temp and voltage)):
        raise Exception('''
            You must supply at least one of these collections of parameters:
                * filepath
                * detector, source, temp, voltage
        ''')

    ### Constructing the path name

    # If supplied, construct the file name from the file name in 'filepath'.
    if filepath:
        filename = os.path.basename(filepath)
        save_path = os.path.splitext(file)
    # Construct the file name from scratch if no 'filepath' is supplied.
    else:
        # Remove all spaces from strings that will be in a file name.
        detector = detector.replace(' ', '')
        source   = source.replace(' ', '')
        temp     = temp.replace(' ', '')
        voltage  = voltage.replace(' ', '')
        etc      = etc.replace(' ', '')

        save_path = f'{detector}_{source}_{temp}_{voltage}'
    
    # Prepend the description if specified
    if description:
        save_path = f'{description}_{save_path}'

    # Append extra info to the file name if specified
    if etc:
        save_path += f'_{etc}'

    # Append the file extension
    save_path += ext

    # Prepend the detector directory if desired
    if sep_by_detector:
        save_path = f'{detector}/{save_path}'

    # Prepend the save directory if specified
    if save_dir:
        save_path = f'{save_dir}/{save_path}'

    return save_path


def gamma_count_map(filepath, save=True, path_constructor=path_constructor,
    save_dir='', sep_by_detector=False, detector='', etc='', ext='.txt'):
    '''
    Generates count map data for raw gamma flood data. No corrections are 
    made for pixel gain here.

    Arguments:
        filepath: str
            The filepath to the gamma flood data

    Keyword Arguments:
        save: bool 
            If True, saves count_map as an ascii file.
            (default: True)
        path_constructor: function
            A function that takes the same parameters as the function
            'gamma.path_constructor' that returns a string representing a 
            path to which the file will be saved.
            (default: gamma.path_constructor)
        etc: str 
            Other important information.
            (default: '')
        save_dir: str
            The directory to which the count_map file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        sep_by_detector: bool
            If True, constructs the file path such that the file is saved in 
            a subdirectory of 'save_dir' named according to the string 
            passed for 'detector'. Setting this to 'True' makes 'detector' a 
            required kwarg, even if 'filepath' is specified.
            (default: False)
        ext: str
            The file name extension for the count_map file. 
            (default: '.txt')

    Return:
        count_map: 2D numpy.ndarray
            A 32 x 32 array of (ints?). Each entry represents the number of
            counts read by the detector pixel at the corresponding index.
    '''
    # Generating the save path, if needed.
    if save:
        save_path = path_constructor(ext, filepath=filepath, 
            save_dir=save_dir, sep_by_detector=sep_by_detector, 
            detector=detector, etc=etc)

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
    count_map = np.empty((32, 32))

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


def plot_pixel_map(count_map, plot_width=550, plot_height=500,
    low=None, high=None, low_color='grey', high_color='red',
     cb_label_standoff=8, cb_title_standoff=12, color_mapping='linear', 
    title='Pixel Map', palette='Viridis256'):
    '''
    Plots a heat map of the counts detected by each pixel.

    If unfamiliar with bokeh:
        To display or save the figure object returned by this function, see
        the documentation for the bokeh.io module at 
        https://bokeh.pydata.org/en/latest/docs/reference/io.html
        The return object of this function can be passed as the parameter 
        labelled 'obj' for any of the functions documented there. For 
        example, to view in a jupyter notebook:
            from bokeh.io import output_notebook, show
            import numpy as np

            count_map = np.loadtxt('count_map_file_name.txt')
            plot = plot_pixel_map(count_map)
            show(plot)

    Arguments:
        count_map: 2D array
            A 32 x 32 array of numbers. Each entry represents the number of
            counts read by the detector pixel at the corresponding index.

    Keyword Arguments:
        plot_width: int
            The width of the plot in pixels
            (default: 550)
        plot_height: int
            The height of the plot in pixels
            (default: 500)
        low: int or float
            The value below which elements of count_map are mapped to the
            lowest color.
            (default: lowest non-zero value in count_map)
        high: int or float
            The value above which elements of count_map are mapped to the
            highest color.
            (default: largest non-zero value in count_map)
        low_color: str or 3-tuple of ints or 4-tuple(int, int, int, float)
            Color to be used if data is lower than low value. If None,
            values lower than low are mapped to the first color in the
            palette. If str, must be either a hex string startig with '#' 
            or a named SVG color. If a tuple, 1st 3 entries are RGB out of
            255, and the 4th optional entry is alpha. For details,
            https://bokeh.pydata.org/en/latest/docs/reference/core/properties.html#bokeh.core.properties.Color
            (default: 'grey')
        high_color: str or 3-tuple of ints or 4-tuple(int, int, int, float)
            Color to be used if data is higher than 'high' value. If None, 
            values higher than 'high' are mapped to the last color in the 
            palette. See 'low_color' for help formatting this parameter.
            (default: 'red')
        cb_label_standoff: int
            The number of pixels by which the color bar's ticker labels
            are offset from the color bar.
            (deault: 8)
        cb_title_standoff: int
            The number of pixels the color bar's title is above the color bar
            (default: 12)
        title: str
            The title displayed on the plot.
            (default: 'Pixel Map')
        palette: str or sequence
            A sequence of colors to use as the target palette for mapping.
            This property can also be set as a String, to the name of any 
            of the palettes shown in bokeh.palettes. For example, you could
            also set the palette to 'Inferno256', 'Magma256', or 'Plamsa256'.
            (default: 'Viridis256')

    Return:
        A bokeh.plotting.Figure object with a heat map of 'count_map'
        plotted. 
    '''

    # Put data in a ColumnDataSource object such that data will display
    # correctly upon hovering over a pixel.
    source = bm.ColumnDataSource(data={
        'x': [0.5],
        'y': [0.5],
        'counts': [count_map]
    })

    # Format of the tooltip when hovering over a pixel.
    tooltips = [
        ('(x, y)', '($x{g}, $y{g})'),
        ('Counts', '@counts{g}')
    ]

    # Generates the figure canvas 'p'
    p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height,
        x_range=(0.5, 32.5), y_range=(0.5, 32.5),
        tools='pan,wheel_zoom,box_zoom,save,reset,hover',
        tooltips=tooltips, toolbar_location='above', title=title)

    # Set heatmap axes to have tick intervals scale by a factor of 2.
    # This causes the 32nd pixel to have its own tick. 
    axis_ticker = bm.AdaptiveTicker(max_interval=None, min_interval=1, 
        num_minor_ticks=0, mantissas=[2], base=2)
    p.xaxis.ticker = axis_ticker
    p.yaxis.ticker = axis_ticker

    # If not specified, assign values to 'high' and 'low' based on data.
    if (not low) or (not high):
        flat_counts = count_map.flatten()
        flat_counts = np.ma.masked_values(flat_counts, 0)
    if not low:
        low = np.amin(flat_counts)
    if not high:
        high = np.amax(flat_counts)

    # Formatting the color bar
    if color_mapping == 'linear':
        color_mapper = bm.LinearColorMapper(palette=palette, 
        low=low, high=high, low_color=low_color, high_color=high_color)
    elif color_mapping == 'log':
        color_mapper = bm.LogColorMapper(palette=palette,
            low=low, high=high)

    cb_ticker = bm.AdaptiveTicker()

    color_bar = bm.ColorBar(location=(0, 0), ticker=cb_ticker,
        label_standoff=cb_label_standoff, color_mapper=color_mapper,
        title='Counts', title_standoff=cb_title_standoff)

    p.add_layout(color_bar, 'right')

    # Generates the heatmap itself
    p.image(source=source, image='counts', x='x', y='y', dw=32, dh=32,
        color_mapper=color_mapper)

    return p


def plot_count_hist(count_map, bins=100, plot_width=600, plot_height=400,
    title='Count Histogram'):
    '''
    Plots a count histogram with respect to the pixels.

    Arguments:
        count_map: 2D array
            A 32 x 32 array of numbers. Each entry represents the number of
            counts read by the detector pixel at the corresponding index.

    Keyword Arguments:
        bins : int or sequence of scalars or str, optional
            If `bins` is an int, it defines the number of equal-width
            bins in the given range (10, by default). If `bins` is a
            sequence, it defines the bin edges, including the rightmost
            edge, allowing for non-uniform bin widths.
            For information about str values of 'bins', see the numpy
            documentation of the parameter with 'help(np.histogram)'.
        plot_width: int
            The width of the plot in pixels
            (default: 550)
        plot_height: int
            The height of the plot in pixels
            (default: 500)
        title: str
            The title displayed on the plot.
            (default: 'Count Histogram')
    '''
    # Binning the data
    counts = count_map.flatten()
    hist, edges = np.histogram(count_map, bins=bins)

    # Generating the Figure object 'p'
    p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height,
        title=title, x_axis_label='Counts', y_axis_label='Number of Pixels')

    # Plotting rectangular glyphs for bins.
    p.quad(left=edges[:-1], right=edges[1:], top=hist, bottom=0)

    return p


def quick_gain(filepath, source, path_constructor=path_constructor, 
    plot=True, plot_dir='', plot_ext='.eps', save_data=True, data_ext='.txt',
    data_dir='', detector='', temp='', voltage='', etc=''):
    '''
    Generates gain correction data from the raw gamma flood event data.
    Arguments:
        filepath: str
            The filepath to the gamma flood data
        source: str
            The radioactive source used. Can be either 'Am241' or 'Co57'.

    Keyword Arguments:
        plot: bool
            If true, plots and energy spectrum for each pixel
        save: bool 
            If True, saves count_map as a .txt file, and a non-empty string
            must be supplied to the 'detector', 'source', 'temp', and 
            'voltage' kwargs. If False, then nothing is saved, and these
            parameters may be left unspecified.
            (default: True)
        detector: str
            The detector ID
            (default: '')
        temp: str
            The temperpature
            (default: '')
        voltage: str
            The voltage
            (default: '')
        etc: str 
            Other important information
            (default: '')
        save_dir: str
            The directory to which the count_map file will be saved. If left
            unspecified, the file will be saved to the current directory.
            (default: '')
        ext: str
            The file name extension for the count_map file. 
            (default: '.txt')

    Return:
        gain: 2D numpy.ndarray
            A 32 x 32 array of (ints?). Each entry represents the number of
            counts read by the detector pixel at the corresponding index.
    '''

    if save_data:
        data_path = path_constructor(filename=filename, ext=data_ext,
            description='gain')

    if plot:
        plot_path = path_constructor(filename=filename, description='gain')

    # From http://www.nndc.bnl.gov/nudat2/indx_dec.jsp
    lines = {
        'Am241': 59.54,
        'Co57': 122.06
    }

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
                # TODO
                # Why only take argmax of slice from 3000 to 6000 channels?
                # Should this be a parameter to the function or hard-coded?
                # 'centroid' is the channel with the most counts
                centroid = np.argmax(spectrum[3000:6000]) + 3000
                # TODO
                # Same here. Should the range of fit channels be a tunable
                # parameter?
                fit_channels = np.arange(centroid - 100, centroid + 200)
                g_init = models.Gaussian1D(amplitude=spectrum[centroid], 
                    mean=centroid, stddev = 75)
                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, fit_channels, spectrum[fit_channels])

                # If we can determine the covariance matrix (which implies
                # that the fit succeeded), then calculate this pixel's gain
                if fit_g.fit_info['param_cov'] is not None:
                    gain[y, x] = lines[source] / g.mean
                    if plot:
                        plt.figure()
                        sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
                        fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
                        mean_err = np.diag(fit_g.fit_info['param_cov'])[1]
                        frac_err = np.sqrt(np.square(fwhm_err) 
                            + np.square(g.fwhm * mean_err / g.mean)) / g.mean
                        str_err = str(int(round(
                            frac_err * lines[source] * 1000
                        )))
                        str_fwhm = str(int(round(
                                lines[source] * 1000 * g.fwhm/g.mean, 0
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


def gain_correct(filepath, gain):
    '''
    Applies gain correction to the data to obtain energy data for events.
    '''
    pass

# For testing, will plot a pixel map and histogram for a file of 
# count map data.
if __name__ == '__main__':
    count_map = np.loadtxt('count_map_H100_Am241_-10C_0V.txt')
    pixel_map = plot_pixel_map(count_map)
    histogram = plot_count_hist(count_map)

    bokeh.io.output_file('count_map.html')
    bokeh.io.show(pixel_map)
    bokeh.io.show(histogram)