# gAia
## Work Package 4 - Arbeitspaket 4
AI unterstützte Datenfusion und Qualitätskontrolle. 
This repository contains the code for the work package 4 of the gAia project.
This work package has the goal of providing an alternative model to the one in WP3 for Landslide detection on top of Satellite imagery. 
The model takes as input the processed NDVI Sentinel tiles used in WP3, computes new features that extract contextual pixel information, 
and performs Time-Series change-detection analysis in order to detect Landslides.
The 

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

### Data
The whole project works in conjunction with a Linux server that hosts the data.
