import nudetect as nd
from nudetect import Source
import pytest

import pandas as pd

# Instance initialization
s_init = Source('Co57', 1000, 10.0, '2-10-1999', 
    info='initialized using the default constructor')

s_csv_default = Source.from_csv('Am241')

s_csv_specific = Source.from_csv(CIT_number=2037)

# Testing the returns of the instance methods
def test_instance_methods():
    assert s_init.line(14) == 14.41295
    assert s_csv_default.line(14) == 13.8520
    assert s_csv_specific.chan_range(7, lower_bound=400)[0] >= 400


def test_class_methods():
    print('Raw line data from load_line_data: ', Source.load_line_data())
    print('\nData from the print_line_data method: ')
    Source.print_line_data()


def test_csv_modification():
    # Copying the csv source log file for testing
    Source.source_df.to_csv('test_xray_sources.csv')
    # Setting the csv path to this new file
    Source.source_csv_path = 'test_xray_sources.csv'

    s_csv_specific.set_default_source()
    assert s_csv_specific.slice_source_df()[1]['default source'] == True
    assert nd.slice_source_df(1967)[1]['default source'] == False

    s_init.add_source()
    assert 1000 in list(Source.source_df.loc[:, 'CIT number'])

    s_init.modify_source_info(' added info')
    assert s_init.slice_source_df()[1]['info'] != \
        'initialized using the default constructor'

    original_length = len('initialized using the default constructor')
    assert s_init.slice_source_df()[1]['info'][:original_length] == \
        'initialized using the default constructor'

#test_csv_modification()