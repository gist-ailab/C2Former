## Environment Setting

```
conda create -n C2F python=3.8
conda activate C2F

# set cuda version to 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# install mmrotate
pip install openmim
mim install mmcv
mim install mmdet

# install other dependencies
pip install -r build.txt
pip install -r optional.txt
pip install -r runtime.txt
pip install -r tests.txt

```

- versions of mm family
```
pip list | grep mm
```
mmcv-full              1.5.3        (>= 1.5.3,<=1.8.0)  
mmdet                  2.25.1       (>= 2.25.1,<3.0.0)  
mmengine               0.10.4  


```
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMRotate installation
import mmrotate
print(mmrotate.__version__)

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

```



## Execute inference
```
python tools/test.py --config configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py --checkpoint /SSDe/heeseon/src/C2Former/pretrain_weights/resnet50-2stream.pth --out work_dirs/C2Former/results.pkl --gpu-ids 1


```

## Train
```
export PYTHONPATH=/SSDe/heeseon/src/C2Former/mmrotate:$PYTHONPATH


```