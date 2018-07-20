'''
Module with functions for analysis of gamma flood data.
PH: Pulse height
STIM: Mask value. If 1, event is from voltage pulse, not photon. (Usually applied to pixel (10, 10))
UP: micro pluse -- also a mask (in noise)
'''

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting

# Bokeh for interactive plots (v. 0.13.0 at time of writing)
import bokeh.io
import bokeh.plotting
import bokeh.models as bm

def gamma_count_map(filepath, save=True, detector='', source='', temp='',
    voltage='', etc='', save_dir='', ext='.txt'):
    '''
    Generates count map data for raw gamma flood data. No corrections are 
    made for pixel gain here.

    Arguments:
        filepath: str
            The filepath to the gamma flood data

    Keyword Arguments:
        save: bool 
            If True, saves count_map as a .txt file, and a non-empty string
            must be supplied to the 'detector', 'source', 'temp', and 
            'voltage' kwargs. If False, then nothing is saved, and these
            parameters may be left unspecified.
            (default: True)
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
        count_map: 2D numpy.ndarray
            A 32 x 32 array of (ints?). Each entry represents the number of
            counts read by the detector pixel at the corresponding index.
    '''
    # Parameter housekeeping if saving count_map
    if save:
        # Request unsupplied parameters necessary to construct the filename
        # for the count_map data.
        while not detector:
            detector = input('Enter the detector ID (required): ')
        while not source:
            source = input('Enter the source (required): ')
        while not temp:
            temp = input('Enter the temperature (required): ')
        while not voltage:
            voltage = input('Enter the voltage (required): ')

        # Remove all spaces from strings that will be in a file name.
        detector = detector.replace(' ', '')
        source   = source.replace(' ', '')
        temp     = temp.replace(' ', '')
        voltage  = voltage.replace(' ', '')
        etc      = etc.replace(' ', '')

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

    PHmask = 0 < np.array(data['PH'][START:END])
    STIMmask = np.array(data['STIM'][START:END]) == 0
    TOTmask = np.multiply(PHmask, STIMmask)

    # Generate the count_map from event data
    count_map = np.empty((32, 32))

    for i in range(32):
        RAWXmask = np.array(data['RAWX'][START:END]) == i
        for j in range(32):
            RAWYmask = np.array(data['RAWY'][START:END]) == j
            count_map[i, j] = np.sum(np.multiply(
                TOTmask, np.multiply(RAWXmask, RAWYmask)))

    # Saves the 'count_map' array as a human-readable text file.
    if save:
        # Construct the file name
        save_path = f'count_map_{detector}_{source}_{temp}_{voltage}'
        if etc:
            save_path += f'_{etc}'
        save_path += ext
        # Prepend the save directory if specified
        if save_dir:
            save_path = f'{save_dir}/{save_path}'
    
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


def quick_gain(filepath):
    '''
    Generates a gain correction file from the gamma flood data.
    '''
    pass

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