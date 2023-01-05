# Labeling

## 1. Setup
Run the following command:
```shell script
pip install numpy Pillow opencv-python tqdm geopy tifffile xmltodict matplotlib requests cartopy
```
Download Maperitive from [here](http://maperitive.net/download/Maperitive-latest.zip) and extract its content under your ```<PROJECT_ROOT>``` directory.

If you have troubles installing Cartopy, you can also try to install it manually by:

```shell script
git clone https://github.com/SciTools/cartopy.git
cd cartopy
# Uncomment the following to specify non-standard include and library paths
# python setup.py build_ext -I/path/to/include -L/path/to/lib
python setup.py install
```

## 2. Data and folder structure
After downloading the dataset and setting up the labeling part, your folder structure should look similar to this:
```
PROJECT_ROOT
├── DATA_ROOT
│   ├── morges 
|   ├── cully
|   ├── neuchatel
|   ├── vevey
|   ├── montreaux
|   ├── human_drone_ann_labels
|   ├── CUSTOM_IMG_DIR
|   └── ...  
├── Maperitive
|   ├── Maperitive.exe
|   └── ...  
└── labeling
|   ├── overlap.py
|   ├── label.py
|   ├── custom.mrules
|   ├── organise.py
|   └── ...  
...     
```
You should create a folder with the desired drone images that you want to use for further training and validation (for example: ```CUSTOM_IMG_DIR```), and define
a list of validation image names. You can find our default list under ```val.txt```.

 **NOTE**: You need both .JPG and .DNG files for each sample (this is code-checked).

## 3. Training-validation images overlap

The training and validation images that you choose may overlap. Therefore, by running the following command,
you can get rid of training images that overlap with a predefined list of validation images, and directly create the splits ```.txt``` files
under ```split_dir_name```:
```shell script
cd labeling
python overlap.py --dataset_root <relative_path_to_data_root> --images_dir <custom_img_dir_name> --splits_dir <split_dir_name>
```

Example:
```shell script
cd labeling
python overlap.py --dataset_root ../DATA_ROOT --images_dir CUSTOM_IMG_DIR --splits_dir splits
```

## 3. Create satellite images, raster images or satellite labels
Run the following command, for obtaining satellite images, raster images and satellite labels:
```shell script
cd labeling
python label.py --images_dir_path <relative_path_to_custom_img_dir> --rules custom.mrules --maperitive <relative_path_to_Maperitive_exe_file> --satellite --label --raster --access_token <token_value>
```

In the above command, you can choose any of the ```--satellite```, ```--label``` or ```--raster``` command
line arguments, depending on what you want to generate. 

Example:
```shell script
cd labeling
python label.py --images_dir_path ../DATA_ROOT/CUSTOM_IMG_DIR --rules custom.mrules --maperitive ../Maperitive/Maperitive.exe --satellite --label --access_token XYZ
```

**NOTE1**: You also need to specify the ```--access_token``` argument.
This argument is the MapBox access token, used for satellite images queries. To obtain this token create an
account on [Mapbox](https://www.mapbox.com/) and when logged-in go to ```Tokens``` tab and copy the default public token given.

**NOTE2**: ```custom.mrules``` is our custom rule file used by Maperitive. More about this at [Maperitive](http://maperitive.net/).

After the script finishes, you will have all images under ```CUSTOM_IMG_DIR```.

## 4. Folder structure and image names used for training and validation

For the rest of the project, the folder structure should be changed in the following way:
```
PROJECT_ROOT
├── data
│   ├── DATASET_NAME                     
│   │   ├── ann_dir              <- satellite labels
|   |   |   ├── <img_name_1>.png     
│   │   │   └── ...
│   │   ├── img_dir              <- satellite images
|   |   |   ├── <img_name_1>.jpg    
│   │   │   └── ...
│   │   ├── human_drone_ann_dir  <- drone labels
│   │   ├── drone_dir            <- drone images
│   │   ├── splits               
│   │   │   ├── train.txt        <- train image names
│   │   │   └── val.txt          <- val image names
│   │   ├── results1             <- other folders with results
|   |   |   ├── <img_name_1>.png   
│   │   │   └── ...
│   │   └── ...   
├── labeling
├── Maperitive
├── DATA_ROOT               
└── ...
```
Your ```CUSTOM_IMG_DIR``` will be transformed into what is under the ```data``` folder, in the above schema.
You can create this directory structure by running the following function:
````shell script
cd labeling
python organise.py --old_dataset_root <relative_path> \
                   --old_images_root <dir_name> \
                   --new_dataset_name <dir_name> \
                   --data_root <dir_name> \
                   --img_dir_name <dir_name> \
                   --ann_dir_name <dir_name> \
                   --drone_dir_name <dir_name>
````

Example:
````shell script
cd labeling
python organise.py --old_dataset_root ../DATA_ROOT \
                   --old_images_root CUSTOM_IMG_DIR \
                   --new_dataset_name DATASET_NAME \
                   --data_root ../data \
                   --img_dir_name img_dir \
                   --ann_dir_name ann_dir \
                   --drone_dir_name drone_dir
````

You can upload the ```data``` folder, under the ```PROJECT_ROOT``` folder, on Google Drive, for running experiments on Google Colab.

## 5. Further information and notes about data

Every label and prediction, stored in the corresponding folders, should be ```PIL``` images converted to mode 'P', with a specific pallette added. This means that only categorical 
integer values will be stored for each pixel, depending on the number of classes in image, and for each categorical value we will have a coressponding RGB value, according to our defined pallette.
For example, for this categorical dummy label (lets say ```img1.png```) :

[[```0```, ```1```], 

 [```2```, ```3```]]


 with 4 pixels from the following categories: 'background'(0), 'building'(1), 'road'(2), 'water'(3) and the following pallette:
`````[[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]]`````, we will know the following correspondences:
backgound - black, building - red, road - blue, water - green.

**IMPORTANT**: Check that the human annotated drone labels have the same format. Check the function ```change_human_drone_anns()``` from ```organise.py```.

Images from our dataset are satellite and drone images of 5280 x 3956 pixels. There are 4 classes in total, i.e. 'background', 'building', 'road' and 'water'.

