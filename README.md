# MFD

Project for masked/non masked face detection research.

## Project and data preparation.

a. Clone repository.

```shell
git clone --recurse-submodules https://github.com/DenDiv/vedadet.git
```

b. Download [eval_tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip) and unzip it to the root of repository. 

c. Download [WIDER](http://shuoyang1213.me/WIDERFACE/) train/val datasets. Also download [MAFA](https://www.kaggle.com/rahulmangalampalli/mafa-data). Extract both datasets as in the image below:
```plain
vedadet
├── vedadet
├── vedacore
├── tools
├── configs
├── data
│   ├── WIDERFace
│   │   ├── WIDER_train
│   │   │   ├── 0--Parade
│   │   │   ├── ......
│   │   │   ├── 61--Street_Battle
│   │   ├── WIDER_val
│   │   │   ├── 0--Parade
│   │   │   ├── ......
│   │   │   ├── 61--Street_Battle
│   ├── MAFA
│   │   ├── MAFA_train
│   │   │   ├── images
│   │   ├── MAFA_test
│   │   │   ├── images
├── ......
├── eval_tools
├── wider-face-pascal-voc-annotations
├── MAFA_anno.zip
├── README.md
```

d. Prepare annotations for train/val WIDER and train/test MAFA.

```shell
./prepare_anno.sh
```
After the last line WIDER_unmasked_anno - directory with xmls annotations and train_unmasked.txt (val_unmasked.txt) - file with image names should be created in WIDER_train (WIDER_val) directory;
MAFA_anno and train.txt (test.txt) in MAFA_train (MAFA_test).

Comment: for generating WIDER PASCAL VOC annotations was initially used [this](https://github.com/akofman/wider-face-pascal-voc-annotations) project, for MAFA - FMLD_annotations.zip was downloaded from [this](https://github.com/borutb-fri/FMLD) research but with deleting WIDER 
annotations in .zip (MAFA_anno.zip in current repo).

Following [this](https://github.com/borutb-fri/FMLD) research faces in .xml annotation-files can be one out of three types: *masked_face*, *unmasked_face* and *incorrectly_masked_face*.

## Install vedadet subproject.

You can also follow [this](https://github.com/Media-Smart/vedadet) instruction. It's all the same here.

### Requirements

- Linux
- Python 3.7+
- PyTorch 1.6.0 or higher
- CUDA 10.2 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 20.04.2 LTS
- CUDA: 11.5
- PyTorch 1.10.0
- Python 3.8.5

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedadet python=3.8.5 -y
conda activate vedadet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Install vedadet.

```shell
vedadet_root=${PWD}
pip install -r requirements/build.txt
pip install -v -e .
```

## Train

Train on MAFA+WIDER dataset to predict *masked_face*, *unmasked_face* and *incorrectly_masked_face*. Weights, optimizer
and meta files will be generated in `workdir`.

b. Multi-GPUs training
```shell
tools/dist_trainval.sh configs/trainval/tinaface_masked/tinaface_r50_fpn_bn_wider_mafa.py "0,1"
```

c. Single GPU training
```shell
CUDA_VISIBLE_DEVICES="0" python tools/trainval.py configs/trainval/tinaface_masked/tinaface_r50_fpn_bn_wider_mafa.py
```

## Evaluation

Evaluation is taking place for WIDER_val and MAFA_test separately. Widerface txt files will be generated at `--outdir` 
after running the following code and then use [eval_tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip) 
to evaluate the WIDERFACE performance:
```shell
CUDA_VISIBLE_DEVICES="0" python configs/trainval/tinaface/test_widerface.py configs/trainval/tinaface_masked/tinaface_r50_fpn_bn_wider_eval.py weight_path.pth --outdir eval_dirs/tmp/wider/
```

For MAFA following code will generate `mafa_test_pr.png` file with pr curves:
```shell
CUDA_VISIBLE_DEVICES="0" python configs/trainval/tinaface_masked/test_mafa.py configs/trainval/tinaface_masked/tinaface_r50_fpn_bn_mafa_eval.py weight_path.pth --outdir eval_dirs/tmp/mafa/
```

## Inference 

Draw bbxs on images or video (for video works slow currently, optimization in plans). Firstly change weight file path in your *config.py* file.

For image:
```shell
CUDA_VISIBLE_DEVICES="0" python tools/infer.py configs/infer/tinaface_masked/tinaface_r50_fpn_bn_wider_mafa.py image_path
```

For video (output.avi - your video), output_with_bbxs.avi will be created:
```shell
CUDA_VISIBLE_DEVICES="0" python MFD/webcam_video_capture/add_bbxs.py configs/infer/tinaface_masked/tinaface_r50_fpn_bn_wider_mafa_video.py MFD/webcam_video_capture/videos/output.avi
```