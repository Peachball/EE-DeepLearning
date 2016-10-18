# EE Code
This repository holds code used to generate the data used in Chen Xu's EE for
the IB diploma.

## Requirements
Although the requirements will be listed below, it is heavily recommended that
you use [Anaconda](https://www.continuum.io/downloads) to manage the
depedencies.

* Python (2.7 or 3.5)
* Numpy
* Scipy
* Theano (0.8 or above)
* Keras (1.1)
* Matplotlib

## Running the Code
Although there are a lot of scripts in the folder, most of them are test
implementations of other algorithms. To run the model, you merely need to run
the MusicLoader script:
```bash
python MusicLoader.py
```
and then the data will be generated into the "data" folder

To view the data, you simply need to run:
```bash
python data_vis.py
```
and they will be displayed as windows
