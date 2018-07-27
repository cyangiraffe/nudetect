# X-ray Vision

## Getting Started


### Prerequisites

* Numpy - data is stored in numpy arrays and in files that are best read by numpy (code written in v. 1.14)
* Bokeh - plotting is done in Bokeh to facilitate interactivity (code written in v. 0.13)

### Installing

Currently, the package can be crudely installed by putting the 'gamma.py' file into the directory in which you want to run data analysis scripts and then imported like any other module. An example script is below.
```python
import numpy as np
import gamma
from bokeh.io import output_file, show

output_file('count_map.html')

count_map = gamma.count_map('20170315_H100_gamma_Am241_-10C.0V.fits')

p = gamma.bokeh_pixel_map(count_map, value_label='Counts',
	low=4000, high=30000)

show(p)
```

## Built With

* Astropy (v. 3.0)
* Numpy (v. 1.14)
* Bokeh (v. 0.13)

## Authors

* **Hiromasa Miyasaka** - *Wrote original IDL scripts*
* **Sean Pike** - *Wrote scripts in Python* - [snpike](https://github.com/snpike/)
* **Julian Sanders** - *Documentation, organization, and incorporating Bokeh plots* - [colcaboose](https://github.com/colcaboose)
* **Andrew Sosanya** - *Documentation and logistics* - [DrewSosa](https://github.com/DrewSosa)