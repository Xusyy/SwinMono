# SwinMono: Swin Transformer-based Monocular Depth Estimation on SeasonDepth

## Dataset
We perform our experiments on [[SeasonDepth]](https://github.com/SeasonDepth/SeasonDepth).


## Environment
Tested on 
```
python==3.7.7
torch==1.11.0
h5py==3.6.0
scipy==1.7.3
opencv-python==4.5.5
mmcv==1.5.0
timm=0.5.4
albumentations=1.1.0
tensorboardX==2.5
gdown==4.4.0
```
You can install above package with 
```
$ pip install -r requirements.txt
```


## Test

Download our pretrained model [link](https://drive.google.com/file/d/10pKJn9hO4sI5XDNqTHIoCQuT3D_4eENc/view?usp=sharing) and put it in './ckpt/' folder.
  ```
  $ python ./code/test.py --data_path <dir_to_data> --save_eval_pngs
  ```
  
## Train

Download the pretrained model 'swin_base_patch4_window7_224_22k.pth' provided by [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and put it in './ckpt/' folder.
  ```
  $ python ./code/train.py --data_path <dir_to_data> --exp_name train --save_model --save_result
  ```
 


## License
For non-commercial purpose only (research, evaluation etc). 


## References

[1] Global-Local Path Networks for Monocular Depth Estimationwith Vertical CutDepth. [[code]](https://github.com/vinvino02/GLPDepth)

[2] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. [[code]](https://github.com/microsoft/Swin-Transformer)
