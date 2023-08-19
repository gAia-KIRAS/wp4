# gAia
## Work Package 4 - Arbeitspaket 4
AI unterstützte Datenfusion und Qualitätskontrolle. 
This repository contains the code for the work package 4 of the gAia project.
This work package has the goal of providing an alternative model to the one in WP3 for Landslide detection on top of Satellite imagery. 
The model takes as input the processed NDVI Sentinel tiles used in WP3, computes new features that extract contextual pixel information, 
and performs Time-Series change-detection analysis in order to detect Landslides. 

### Packages and dependencies
The code is written in Python 3.10.11.
All package dependencies are listed in the requirements.txt file. They can be installed with the following command:
```
pip install -r requirements.txt
```
The `gdal` installation can fail via pip and the installation via conda is recommended:
```
conda install -c conda-forge gdal=3.0.2
```

### Linux server
The whole project works in conjunction with a Linux server that hosts the data.
For access to the server, contact the project supervisor.

### Data
The landslide detection made on this project uses Sentinel-2 imagery. In particular, images from 2018 to 2022.
Sentinel-2 images are divided into tiles, and the 4 tiles that cover the Area of Interest (AOI) of 
the project are used, namely: 33TUM, 33TUN, 33TVM, 33TVN. The change detection and landslide detection is done
on each tile individually.

### Modules and image types
The project is divided into different modules, each one implementing a 'step' in the pipeline. A 'step' is
defined as the process of taking an input image of a defined type A, performing some operations on it, 
and outputting a new image of type B. The image types are 'RAW', 'NCI'.

The modules are:
- `nci`: Performs the step RAW to NCI. It calculates the time-series of NCI images for the RAW images.


Note: `crop` step was initially an individual module. It was later merged into the `nci` module and is now deprecated.
We leave it in the repository for reference.

#### NCI
We start with the NDVI-reconstructed time series, which are NDVI time-series that have been corrected for clouds and shadows.
For each date 