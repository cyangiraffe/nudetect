# X-ray Vision

## Getting Started


### Prerequisites

* Numpy - data is stored in numpy arrays and in files that are best read by numpy (code written in v. 1.14)
* Bokeh - plotting is done in Bokeh to facilitate interactivity (code written in v. 0.13)

### Installing

Currently, the package can be crudely installed by putting the 'gamma.py' file into the directory in which you want to run data analysis scripts and then imported like any other module. An example script one might run from a jupyter notebook is as follows.
```
import numpy as np
import gamma
from bokeh.io import output_notebook, show

output_notebook()

count_map = gamma_count_map('20170315_H100_gamma_Am241_-10C.0V.fits',
                    detector='H100', source='Am241', temp='-10C',
                    voltage='0V')

p = gamma.plot_pixel_map(count_map, low=4000, high=30000)

show(p)
```

## Built With

* Astropy (v. 3.0.1)
* Numpy (v. 1.14.5)
* Bokeh (v. 0.13.0)

## Authors

* **Sean Pike** - *Initial script writing* - [snpike](https://github.com/snpike/)
* **Julian Sanders** - *Editing scripts and incorporating interactive plots* - [colcaboose](https://github.com/colcaboose)

## License

Do we have a license for this? Should we?? What's a license???