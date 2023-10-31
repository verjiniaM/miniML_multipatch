<img align="left" width="100" height="100" src="minML_icon.png">

# miniML: A deep learning framework for synaptic event detection

[![minimal Python version](https://img.shields.io/badge/Python%3E%3D-3.9-grey.svg?style=for-the-badge&logo=python&labelColor=3670A0&logoColor=white)](https://www.anaconda.com/download/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-TBD-grey.svg?style=for-the-badge&labelColor=green&logoColor=white)](https://www.dx.doi.org)


This repository contains the code described in the following publication:  
O'Neill P.S., Baccino Calace M., Rupprecht P., Friedrich R.W., Müller, M., and Delvendahl, I. 
(2023) Deep learning-based detection of spontaneous synaptic events. _bioRxiv_ XXX ([doi:xxx](https://tbd))  


### 🧠 ABOUT

miniML is a deep-learning-based tool to detect synaptic events in 1d timeseries data. It uses a CNN-LSTM network architecture that was trained using a large dataset of synaptic events. 

In this repository, we provide pretrained models and Python code to run model inference on recorded data. In addition, an application example is included.

### 💻 INSTALLATION

To use miniML, clone the GitHub Repositiory and install the requirements. Python dependencies are: sklearn, matplotlib, h5py, pandas, numpy, scipy, tensorflow, pyabf. To install all dependencies, run 

`pip install -r requirements.txt`


### ⏱ RUNNING MINIML

First, a miniML *trace* object needs to be created containing 1d timeseries data. Currently, miniML supports loading from HEKA .dat files, Axon .abf files as well as HDF .h5 files. Other files formats need to be imported as Python objects.

Next, a miniML *detection* object can be created, specifying a miniML *model* file to use as well as the *trace* object to operate on. 

Finally, model inference can be run using the *detect_events()* method. 


### 💡 EXAMPLE

The folder "Example/" contains an example recording from a MF–GC synapse together with a commented Jupyter Notebook illustrating the use of miniML.



### ✉️ CONTACT
philippsean.oneill@uzh.ch or igor.delvendahl@uzh.ch