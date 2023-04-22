# learnpytorch

## ref
https://pytorch.org/tutorials/

pip3 install torch torchvision torchaudio

# dep

```
install pytorch first

install yolo dep:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

install image label tool:
pip3 install labelImg  -i https://mirrors.aliyun.com/pypi/simple/
```

## yolo  train

```
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

## yolo eval
```
look at yolo.py
```
