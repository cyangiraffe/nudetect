'''
A testing module for gamma.py
'''

import bokeh.io
import numpy as np
import gamma
import pytest

filepath = '20170315_H100_gamma_Am241_-10C.0V.fits'


def test_Line():
    assert gamma.am.source == 'Am241'
    assert gamma.co.energy == 122.06

# TODO 
# Implement temporary directories in this test. This is commented out
# because you need to set up certain directories for the test to work.

# def test_construct_path():
#     assert gamma.construct_path('hello') == 'hello'

#     assert gamma.construct_path('hello', etc='world',
#         description='desc', detector='foo', ext='html') \
#     == 'desc_hello_world.html'

#     assert gamma.construct_path(filepath + '\n', save_dir='.git')\
#     == '.git/20170315_H100_gamma_Am241_-10C_0V'

#     assert gamma.construct_path(filepath, sep_by_detector=True, 
#         detector='H100')\
#     == 'H100/20170315_H100_gamma_Am241_-10C_0V'

#     assert gamma.construct_path(filepath, save_dir='hello', 
#         sep_by_detector=True, detector='H100')\
#     == 'hello/H100/20170315_H100_gamma_Am241_-10C_0V'

#     with pytest.raises(Exception):
#         gamma.construct_path(filepath, sep_by_detector=True)

#     with pytest.raises(ValueError):
#         gamma.construct_path(filepath, save_dir='hello', 
#             sep_by_detector=True,detector='world')

def test_count_map():
    count_map = np.loadtxt('count_map_H100_Am241_-10C_0V.txt')
    assert (gamma.count_map(filepath) == count_map).all()

def test_quick_gain():
    gain = np.loadtxt('gain_20170315_H100_gamma_Am241_-10C_0V.txt')
    assert (gamma.quick_gain(filepath, gamma.am, detector='H100')
        == gain).all()

def test_spectrum():
    gain = np.loadtxt('gain_20170315_H100_gamma_Am241_-10C_0V.txt')
    spectrum = gamma.get_spectrum(filepath, gain)
    gamma.plot_spectrum(spectrum, gamma.am)