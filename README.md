# gAia: Work Package 4
## AI unterstützte Datenfusion und Qualitätskontrolle. 
This repository contains the code for the work package 4 of the 
[gAia](https://www.sba-research.org/research/projects/gaia/) project.
This work package has the goal of providing an alternative model to the one in WP3 for Landslide detection
on top of Satellite imagery. The model takes as input the processed NDVI Sentinel tiles used in WP3,
computes new features that extract contextual pixel information, 
and performs change analysis to predict the probability of a landslide. 

## Project overview

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
If further issues arise, please refer to [this solution](https://stackoverflow.com/questions/44005694/no-module-named-gdal)

### Linux server
The whole project works in conjunction with a Linux server that hosts the data.
For access to the server, contact the project supervisor. 
If you want to use the project on different data, a server (or local machine) can be set up with the structure
described in the section [folder structure](#folder-structure).

### Data
The landslide detection made on this project uses Sentinel-2 imagery. In particular, images from 2018 to 2022.
Sentinel-2 images are divided into tiles, and the 4 tiles that cover the Area of Interest (AOI) of 
the project are used, namely: 33TUM, 33TUN, 33TVM, 33TVN. The change detection and landslide detection is done
on each tile individually. The original Sentinel-2 images are downloaded from the [ESA website](https://scihub.copernicus.eu/). 

### Modules and image types
The project is divided into different modules, each one implementing a 'step' in the pipeline. A 'step' is
defined as the process of taking an input image of a defined type A, performing some operations on it, 
and outputting a new image of type B. Some additional modules do not output images, but implement the evaluation
or the model training.

The different **image types** are:
- `RAW`: The original Sentinel-2 images, in the format they are downloaded from the ESA website. We consider the 
NDVI_reconstructed products, which are the NDVI images that have been corrected for clouds and shadows. These are 
computed in WP3, and we refer to the [WP3 repository](https://github.com/gAia-KIRAS/wp3) for more information. It only 
contains one band:
  - Band 0: reconstructed NDVI values.
- `NCI`: Neighborhood Correlation Images. These are images that contain four additional features for each pixel,
which are the correlation, the slope, and intercept of a linear model fitted in the change of two consecutive 
neighborhoods (more information in the [NCI module](#nci-module) section).
  - Band 0: neighborhood correlation values.
  - Band 1: neighborhood slope values.
  - Band 2: neighborhood intercept values.
  - Band 3: pixel-vs-neighborhood difference values
- `DELTA`: Delta is the relative change between two NCI images. It is computed as the difference between two NCI images 
divided by the value of the first image. It also includes the relative change in the RAW NDVI images.
  - Band 0: relative change in the correlation values.
  - Band 1: relative change in the slope values.
  - Band 2: relative change in the intercept values.
  - Band 3: relative change in the pixel-vs-neighborhood difference values.
  - Band 4: relative change in the RAW NDVI values
- `CPROB`: Change probability images. These images are the output of the change detection model. They contain the
probability of a landslide in each pixel. They have a single band.
  - Band 0: landslide probability values.

Two extra images types exist but are not used in the main pipeline:
- `CROP`: The raw images cropped according to the region of interest Carinthia. However, this step is now deprecated
because the NCI computation already crops the images.
- `TESTING`: Used for testing purposes.

The **basic modules** perform the main steps of the pipeline. They inherit from the abstract class `Module` 
in `abstract_module.py`. The basic modules are:
- `NCI`: Performs the step RAW to NCI. It calculates the time-series of NCI images for the RAW images. More details in
the [NCI module](#nci-module) section.
- `ChangeComputation`: Performs the step NCI to DELTA. It calculates the time-series of DELTA images.
- `ChangeDetection`: Performs the step DELTA to CPROB. It calculates the time-series of CPROB images and the final
CPROB image.
- `Evaluation`: Evaluates the change detection model. More details in the [Evaluation module](#evaluation-module) section.

Some **additional modules** perform additional tasks:
- `BuildTrainDataset`: Trains the logistic regression model used in the change detection step.
- `IntersectAOI`: Crop RAW images. No longer used, but we add it for completion. The module was used to crop some images
and we used the information to perform an automatic crop in the NCI step.
- `Model fusion`: Uses the weather model to predict on the change detection results.

Finally, `log_reg_aggregation.py` is a script that builds the Logistic Regression model. It is not a module, and should 
not be run again unless the model needs to be retrained.


### Classes and data structures
Some important classes and data structures are defined in the project:
- ``Config`` is the class that contains the configuration parameters of the modules. It is defined in ``config.py``.
- ``IO`` is the class that manages the Input/Output operations. It is defined in ``io_manager.py``.
- ``Module`` is the abstract class that defines the basic structure of a module. It is defined in ``abstract_module.py``.
- ``ImageRef`` is the class that represents an image. It is just a reference and does not contain the image itself (`np.ndarray`).

## Folder structure
As explained above, the project works in conjunction with a Linux server that hosts the data.
The code repository should be cloned in the server, as well as in the local machine. 
The folder structures are a bit different in each case. 

#### Server folder structure
The server folder structure is the following:
```
├── wp3
│   └── sentinel2_L2A
│       ├── 2018
│       │   ├── 33TUM
│       │   │   ├── NDVI_raw
│       │   │   ├── NDVI_reconstructed
│       │   │   └── tmp
│       │   ├── 33TUN
│       │   │   ├── NDVI_raw
│       │   │   ├── NDVI_reconstructed
│       │   │   └── tmp
│       │   ├── 33TVM
│       │   │   ├── NDVI_raw
│       │   │   ├── NDVI_reconstructed
│       │   │   └── tmp
│       │   └── 33TVN
│       │       ├── NDVI_raw
│       │       ├── NDVI_reconstructed
│       │       └── tmp
|       ├── (2019, 2020, 2021)
│       └── 2022
│           ├── 33TUM
│           │   ├── NDVI_raw
│           │   ├── NDVI_reconstructed
│           │   └── tmp
│           ├── 33TUN
│           │   ├── NDVI_raw
│           │   ├── NDVI_reconstructed
│           │   └── tmp
│           ├── 33TVM
│           │   ├── NDVI_raw
│           │   ├── NDVI_reconstructed
│           │   └── tmp
│           └── 33TVN
│               ├── NDVI_raw
│               ├── NDVI_reconstructed
│               └── tmp
└── wp4
    ├── code
    │   └── wp4
    │       └── src
    │           ├── config
    │           ├── io_manager
    │           ├── modules
    │           └── notebooks
    ├── cprob
    │   ├── 2018
    │   │   └── 33TUM
    │   │       └── NDVI_reconstructed
    │   ├── 2019
    │   │   └── 33TUM
    │   │       └── NDVI_reconstructed
    │   ├── 2020
    │   │   └── 33TUM
    │   │       └── NDVI_reconstructed
    │   ├── 2021
    │   │   └── 33TUM
    │   │       └── NDVI_reconstructed
    │   └── 2022
    │       └── 33TUM
    │           └── NDVI_reconstructed
    ├── delta
    │   ├── 2018
    │   │   ├── 33TUM
    │   │   │   ├── NDVI_raw
    │   │   │   └── NDVI_reconstructed
    │   │   ├── 33TUN
    │   │   │   ├── NDVI_raw
    │   │   │   └── NDVI_reconstructed
    │   │   ├── 33TVM
    │   │   │   └── NDVI_reconstructed
    │   │   └── 33TVN
    │   │       └── NDVI_reconstructed
    │   ├── (2019, 2020, 2021)
    │   └── 2022
    │       ├── 33TUM
    │       │   ├── NDVI_raw
    │       │   └── NDVI_reconstructed
    │       ├── 33TUN
    │       │   └── NDVI_reconstructed
    │       ├── 33TVM
    │       │   └── NDVI_reconstructed
    │       └── 33TVN
    │           └── NDVI_reconstructed
    ├── inventory
    ├── nci
    │   ├── 2018
    │   │   ├── 33TUM
    │   │   │   ├── NDVI_raw
    │   │   │   └── NDVI_reconstructed
    │   │   ├── 33TUN
    │   │   │   ├── NDVI_raw
    │   │   │   └── NDVI_reconstructed
    │   │   ├── 33TVM
    │   │   │   ├── NDVI_raw
    │   │   │   └── NDVI_reconstructed
    │   │   └── 33TVN
    │   │       ├── NDVI_raw
    │   │       └── NDVI_reconstructed
    │   ├── (2019, 2020, 2021)
    │   └── 2022
    │       ├── 33TUM
    │       │   ├── NDVI_raw
    │       │   └── NDVI_reconstructed
    ├── operation_records
    └── weather_approach
        ├── datasets
        └── models

```
Essentially, there are two main folders in the server: `wp3` and `wp4`. `wp3` contains the data from WP3, which is the RAW
input for the WP4. `wp4` contains the code and the output of the WP4. The code is in the `code` folder, and the output
is in the `nci`, `delta`, and `cprob` folders. The `inventory` folder contains the inventory of the images used in the
project. The `ts` folder contains the time-series of the images used in the project. 
The `operation_records` folder contains the logs of the operations performed in the project, as well as other relevant 
files such as the change detection output, or the train dataset for the logistic regression model. The `weather_approach`
folder contains the code for the weather approach, which is not used in the main pipeline.

#### Local folder structure
The local folder structure is very similar but more natural, because it does not separate wp3 and wp4, and has the code
(in the `src` folder) and the data (in the `data` folder) separated. The following is an overview:

```
├── data
│   ├── cprob
│   ├── delta
│   ├── inventory
│   ├── operation_records
│   ├── raw
│   ├── nci
|   ├── weather
│   └── ts
├── src
    ├── config
    ├── io_manager
    ├── modules
    └── notebooks
├── config.yaml
├── requirements.txt
├── io_config.yaml
└── README.md
```

## Configuration files and parameters
The project uses two configuration files: `config.yaml` and `io_config.yaml`. The first one contains the parameters
of the modules, and the second one contains parameters for the Input/Output manager (paths, username, etc.). 
They should be **located at the same directory as the `src` folder**, i.e. at the root of the project.
The parameters are explained in the comment lines of the files.

#### Main config (`config.yaml`)

This is an example of the main config file. The parameters are explained in the comment lines.

````yaml
paths:    
  data: data          # Name of the data folder

profiling:
  active: True        # Whether to profile the code. Can be True or False
  browser: False      # Whether to open the browser after profiling. Can be True or False

nci:                        # Parameters for the NCI module
  n_size: 3                 # Neighborhood size: 1, 3, 5, 7, 9... (e.g.: 3 means a 3x3 neighborhood around each pixel)
  conv_lib: torch           # Can be [torch, tf]. The library used to compute the convolution. On the server, torch is used

cd:
  cd_id: "fourth_exec_with_log_reg"   # Any name to identify the execution. It is used to continue and execution
  threshold: 0.995                    # Threshold for detection probability in [0,1]
  type: "log_reg"                     # Can be 'basic_mean', 'nci_logic', 'log_reg'. 
                                      # They are different calculation methods for the change probability 

eval:
  cd_id: "fourth_exec_with_log_reg"   # Name of the change detection execution to evaluate
  type: "both"                        # What ground truth landslide inventory to use: points, polygons, both
  baseline_eval: False                # Whether to evaluate the baseline (i.e. random predictions)
  build_train_dataset: False          # Whether to build the train dataset
  take_positives: False               # Whether to take only the positive samples


execute:
  where: server                   # local, server. Can also be: update_server, update_local
  module: cd                      # The module to execute. Can be nci, cd, delta, eval, btds. 
  time_limit: 60                  # Time limit in minutes for the module execution
  filters:                        # If empty list, all images will be processed
    tile: [33TUM]                 # List of tiles to process. Can be empty list
    year: []                      # List of years to process. Can be empty list
    product: [NDVI_reconstructed] # List of products to process. Should be NDVI_reconstructed


````

#### IO config (`io_config.yaml`)
This is an example of the IO config file. The parameters are explained in the comment lines.

````yaml
server:
  name: kronos.ifs.tuwien.ac.at               # Server domain
  username: your_username                     # Username
  password: your_passowrd                     # Password

paths:
  base_server_dir: "/newstorage2/gaia"          # Base directory on the server
  base_local_dir: "data"                        # Base directory on the local machine
  temp_dir: "tmp"                               # Temporary directory on the server  
  server_repo_root: "/newstorage2/gaia/wp4/code/wp4"    # Path to the code repository on the server
  server_python_executable: "/home/salva/miniconda3/envs/wp4_env/bin/python" # Path to the python executable on the server

files:
  aoi_gpkg: "inventory/Projektgebiet_Kaernten.gpkg" # Path to the AOI geopackage file
  aoi_shp: "inventory/Projektgebiet_Kaernten.shp"   # Path to the AOI shapefile
  aoi_shx: "inventory/Projektgebiet_Kaernten.shx"   # Path to the AOI shapefile
  aoi_cpg: "inventory/Projektgebiet_Kaernten.cpg"   # Path to the AOI shapefile
  aoi_dbf: "inventory/Projektgebiet_Kaernten.dbf"   # Path to the AOI shapefile
  aoi_prj: "inventory/Projektgebiet_Kaernten.prj"   # Path to the AOI shapefile
  inventory_gpkg: "inventory/fixed_inventar_kaernten.gpkg"    # Path to the landslide inventory geopackage file
  inventory_shp: "inventory/Massenbewegungen_im_Detail.shp"   # Path to the landslide inventory shapefile
  inventory_poly_shp: "inventory/Rutschungsflaechen.shp"      # Path to the landslide inventory shapefile
  records: "operation_records/records.csv"                    # Path to the operation records file
  records_cd: "operation_records/records_cd.csv"              # Path to the change detection operation records file
  results_cd: "operation_records/results_cd.csv"              # Path to the change detection results file
  all_images:
    raw: "operation_records/all_images_raw.csv"           # Path to the RAW all_images file
    crop: "operation_records/all_images_crop.csv"         # Path to the CROP all_images file
    nci: "operation_records/all_images_nci.csv"           # Path to the NCI all_images file
    delta: "operation_records/all_images_delta.csv"       # Path to the DELTA all_images file

metadata:
  tiles:    # List of tiles
    - 33TUM
    - 33TUN
    - 33TVM
    - 33TVN
  years:    # List of years
    - 2018
    - 2019
    - 2020
    - 2021
    - 2022
  products: # List of products
    - NDVI_raw              # Alias: NDVIraw
    - NDVI_reconstructed    # Alias: NDVIrec
````

**What should be updated in the ``io_config``?**
- `server.user`: Your username in the server
- `server.password`: Your password in the server
- `server.python_executable`: The path to the python executable in the server

## Executing a module
Everything is executed from the `main.py` file, and the selection of the module and 
all the corresponding parameters are set in the `config.yaml` file.

The complete set of images (input and output) is stored in the server. However, the code can also be
run locally and the input images will be downloaded from the server, the output uploaded to the server, 
and finally everything is removed from the local repository. This is useful for testing purposes.
For a complete execution, the code should be run on the server because it saves the time of uploading and downloading
the images. Recall that one image is around 1GB, so it can take a while to upload/download them.

#### Executing locally
Executing locally has no secret, just run the `main.py` file with the parameter `execute.where` set to `local`.
```
python main.py
```

#### Executing on the server
We execute on the server following the steps:
1. Update the code repository on the server:
```
(local)     git push
(server)    git pull
```
2. Set the appropiate run configuration locally in the `config.yaml` file. 
For example, if we want to execute the NCI module, we set the parameter `execute.module` to `nci`.
3. Update the configuration files on the server, as well as the records files. This is done automatically
running main with the parameter `execute.where` set to `update_server`
4. Run the module on the server. Log in to the server, and run the following command:
```
"python_executable_path" "/newstorage2/gaia/wp4/code/wp4/src/main.py" --server_execution
```
We recommend using the `nice` command:
```
nice -n 10 "python_executable_path" "/newstorage2/gaia/wp4/code/wp4/src/main.py" --server_execution
```
You can follow the execution in the same server through the print logs in the terminal. There
is no need to set the `execute.where` parameter to `server` in the `config.yaml` file because
it is overriden by the `--server_execution` flag.
5. Update the configuration files locally, as well as the records files. This is done automatically
running main with the parameter `execute.where` set to `update_local`.


## Operation record files
The operation record files are csv files that contain relevant information about the execution of the modules.
A description of the files:
- `records.csv`: Contains the information of the execution of the modules, i.e. what images have gone through
which steps of the pipeline. It is updated automatically after each execution.
- `records_cd.csv`: Contains the information of the execution of the change detection module. It is updated
automatically after each execution.
- `results_cd.csv`: Contains the results of the change detection module. It is one of the outputs of the change detection
module, and it is updated automatically after each execution.
- `all_images_raw.csv`: Contains the information of all the RAW images.
- `all_images_nci.csv`: Contains the information of all the NCI images.
- `all_images_delta.csv`: Contains the information of all the DELTA images.
- `positives.csv`: Contains the information of the true positive samples from our change detection. Will be used in ModelFusion.
- ``train_inv.csv``: Training samples for the Logistic Regression model
- ``train_features.csv``: Training samples + features for the Logistic Regression model

_Note_: if some type of images are recomputed, the corresponding `all_images` file should be removed. When some module needs
the file, and it does not exist, it will be recomputed.

## NCI module
The NCI module differentiates our approach from WP3. 
NCI was first introduced in the following paper:
- Jungho Im, John R. Jensen. [A change detection model based on neighborhood correlation image analysis and decision tree classification](https://linkinghub.elsevier.com/retrieve/pii/S0034425705002919)


It is a change detection model that uses the correlation between the pixels of two consecutive images to detect changes.
The idea is that if the correlation between two pixels is high, then the pixels are similar and there is no change.
If the correlation is low, then the pixels are different and there is a change. Some additional features that help explain
the kind of change are the slope and intercept of the linear model fitted in the change of two consecutive neighborhoods.

We compute the NCI images by translating the operations to convolutions. Therefore, we can use the tensor operations
of Pytorch to compute the NCI images. The computation steps are the following:
- Assume neighborhood of size ``d``, for example, ``d=3``. This means that we have a 3x3 neighborhood around each pixel. 
- Assume two images ``I1`` and ``I2``. We want to compute the NCI images for these two images.
- The following filter `d x d`, applied to an image with `stride=1`, `padding=same` would give the image of the neighborhood means.
```
A = 1/d^2 [[1, 1, 1],  [1, 1, 1],  [1, 1, 1]]
```
- The mean can be substracted from the image to obtain the centered image. 
- Then we can substract the means from the image to get the centered image, and pixel-wise multiply with the other centered image
- Apply the filter ``d^2 / (d^2 -1) A`` and the result is the correlation image `r`. 
- Image of squared standard deviations can be found by applying the filter ``A`` to the squared centerd image. 
- Standard deviations allow to compute the slope ``a``
- Intercept ``b`` can be computed as ``b = mean(I2) - a * mean(I1)``


## Evaluation module
The evaluation module is used to evaluate the change detection model. It is a class named `Evaluation` 
and it is defined in `src/modules/evaluation.py`. It uses the landslide inventory
to compute the precision of the model. It takes as input, the change detection results, and the landslide inventory.
The change detection results are a file located in ``operation_records/results_cd.csv`` with the following columns:
  - ``version``: version ID of the CD run
  - ``threshold``: threshold used to filter the detected events
  - ``tile``: tile ID
  - ``year``: year of the image
  - ``row``: row index of the pixel
  - ``column``: column index of the pixel
  - ``date``: date of the prediction
  - ``probability``: prediction probability
  - ``timestamp``: timestamp of the detection. When the prediction was executed
  - ``lat``: latitude coordinate of the pixel
  - ``lon``: longitude coordinate of the pixel

It uses the landslide inventories to compute the precision of the model. 
If ``eval.type`` is ``points``, then only the point inventory is used, located in the file `inventory/Massenbewegungen_im_Detail.shp`
If ``eval.type`` is ``polygons``, then only the polygon inventory is used, located in the file `inventory/Rutschungsflaechen.shp`
If ``eval.type`` is ``both``, then both inventories are used.
If using the point inventory, a match is found if the distance between the landslide and the prediction is less than 6m. 
If using the polygon inventory, a match is found if the landslide polygon contains the prediction pixel.

The evaluation module can also compute a **baseline evaluation**. This is set with the parameter ``eval.baseline_eval``.
The baseline evaluation generates a random prediction with the same number of pixels as the change detection results.
The random prediction is then evaluated with the landslide inventory. This is useful to compare the performance of the
change detection model with a random prediction.

The results of the evaluation are printed in the terminal, and have the following format:
```
    --------------------
    Evaluation Results
    --------------------
    True positives: 153
    False positives: 1463478

    Precision: 0.00010453454456758569
    Recall: 0.148 = 70 / 473

   year   tile  tp      fp  precision
   2018  33TUM  41  351300   0.000117
   2019  33TUM  48  262206   0.000183
   2020  33TUM  16  296447   0.000054
   2021  33TUM  17  255428   0.000067
   2022  33TUM  31  298097   0.000104
```

_Note_: this module can be run using the same module script, i.e. running:
```
python evaluation.py
```
The configuration file is the same as in the main pipeline. 


## Weather model and model fusion
One of the initial goals of WP4 was to integrate different models to improve the landslide detection.
One of the models had to be the satellite imagery model (that we reimplemented using NCI features), 
and the other one was the weather model. 

The notebook `notebooks/weather_model.ipynb` contains the code for the weather model. It is not a
module and has been developed differently than the rest of the project. Use it for references on how the weather model
was developed. It includes:
- Data preprocessing
- Feature creation
- Feature selection
- Model training
- Model evaluation

The relevant output of this notebook is the file ``data/weather/models/rf_step30_from2010_filtered_25fs.pkl``, 
which is the best model trained on the weather data. It is a Decision Tree model that given a point in the map and 
a set of features representing the weather conditions, it predicts the probability of a landslide.

The model fusion is done in the module `ModelFusion`. It takes as input the change detection results, and predicts on
them using the weather model. The results are printed in the terminal. Currently, only the prediction on 
the True positives of the change detection is done, but it can be extended to predict on the change detection results.
The idea was to investigate whether the weather model would remove the true positive landslide detections that are
identified by the change detection model. 

_Note_: predicting on the whole dataset can take long. Improvements could be made to the ``calculate_features`` method.


