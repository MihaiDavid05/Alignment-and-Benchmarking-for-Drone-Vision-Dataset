# Patch Match
The algorithm of Patch Match implemented in Python. Adapted from [this](https://github.com/MingtaoGuo/PatchMatch) repo,
it is a method for warping the satellite labels to match the drone images.

## 1.Introduction
This code mainly implements the algorithm of patch match, for details, please see the paper: [PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing](http://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf).

## 2.Setup
Run the following commands:
```shell script
pip install numpy tqdm numba Pillow matplotlib opencv-python
```

## 3.Run

 Example of command for generating our warped labels:
 ```
python PatchMatch.py --data_dir ../DroneVision/drone_dataset/data/ 
                     --sat_dir img_dir 
                     --drone_dir drone_dir 
                     --label_dir ann_dir
                     --output_dir patchmatch_warped
                     --p_size 3
                     --itr 5
                     --new_w 1056
                     --new_h 792
                     --use_my_init
                     --img_type 'drone'
                     --ref_type 'sat'
 
```

The first 5 parameters are input/output related and the next are Patch Match algorithm's parameters.
You can find further explanation of the parameters in the ```get_cli_arg()``` function in ```PatchMatch.py``` script.



