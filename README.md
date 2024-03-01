# Moving Objects in a Scene

We have 3 main folders 
1) `clipseg` -> has the python file for segmention using text prompt
2) `DragonDiffusion` -> has the file for displacing object in a scene
3) `segment-anything`

## Create environments
```
conda env create -f Displace_Object.yml
conda env create -f Segment_Object.yml
conda activate Segment_Object
pip install git+https://github.com/openai/CLIP.git
cd segment-anything
pip install -e .
```
## Download Model weights

```
cd clipseg
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
The above code when run in order will create a `weights` folder with checkpoints

In Task 1: \
For SAM checkpoints use `sam_vit_h_4b8939.pth` \
For CLIPDense checkpoints use `rd64-uni.pth`

Following steps are needed for Task 2
```
cd DragonDiffusion
mkdir models
cd models
wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt
wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/ip_sd15_64.bin
wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/shape_predictor_68_face_landmarks.dat
```
## Task 1

To segement objects in an image i.e conver the queried object woth a red mask run the following.
For Task 1, from main directory:
```
conda activate Segment_Object
```
```
cd clipseg
python run_object_segment.py \
--image <path to image to segment> \
--class_name <name of object to segment> \
--output <name of output file> \
--mask_path <path to store output mask> \
--sam_ckpt <checkpoint of SAM> \
--clipdense_ckpt <checkpoint to CLIPDense> 
```

A example run would be as follows
```
conda activate Segment_Object
cd clipseg
python run_object_segment.py \
--image stool.jpeg \
--class_name "stool" \
--output Segment_Stool.png \
--mask_path Mask_Stool.png  \
--sam_ckpt weights/sam_vit_h_4b8939.pth \
--clipdense_ckpt weights/rd64-uni.pth 
```


## Task 2

IMPORTANT NOTE: \
  To perform Task 2, please perform Task 1 first so as to obtain the image mask (`Mask_Stool.png` from above example) as it is needed in Task 2. \
  The object `--class_name` should correspond to the segmented object in `--mask_path` \

  For example in the following commands use `Mask_Stool.png` got from Task 1 for the `--mask_path` arguement  
  

For Task 2, from main directory:
```
conda activate Displace_Object
```
```
cd DragonDiffusion
python run_object_displace.py \
--image <path to image to segment> \
--class_name <name of object to segment> \
--output <path of output file> \
--mask_path <path to mask obtained from Task 1> \
--x <displacement along x-axis> \
--y <displacement along y-axis>
```

An example run will be as follows:
```
conda activate Displace_Object
cd DragonDiffusion
python run_object_displace.py \
--image stool.jpeg \
--class_name "stool" \
--output Edit_stool.png \
--mask_path Mask_Stool.png \
--x 20 \
--y -10 
```


## Citation 
1) [Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg) 
2) [Segment Anything](https://github.com/facebookresearch/segment-anything) 
3) [Dragon Diffusion](https://github.com/MC-E/DragonDiffusion)