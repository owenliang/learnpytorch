import torch
import torchvision
import os
import numpy as np

# 基于VOC数据集的预训练lraspp语义分割模型 https://pytorch.org/vision/stable/models.html#table-of-all-available-semantic-segmentation-weights
from torchvision.models.segmentation import FCN_ResNet50_Weights,LRASPP_MobileNet_V3_Large_Weights

'''
# 用pytorch来下载VOC数据集到本地,但是不使用它的dataset,我们自己定义一个预处理简单点的dataset
dataset=torchvision.datasets.VOCSegmentation('vocdataset',image_set='train',download=True)
'''

# target每种分类对应的颜色，一共21种分类，对应21种颜色
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    #print(idx.shape)
    #print(colormap2label[idx].shape)
    return colormap2label[idx]

def voc_label2colormap(labelmap):
    '''将像素分类映射回RGB值'''
    flat_labels=torch.reshape(labelmap,(-1,))
    colormap=[]
    for i,label in enumerate(flat_labels):
        colormap.append(VOC_COLORMAP[label])
    return torch.tensor(colormap).reshape(labelmap.shape[0],labelmap.shape[1],3)

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# 读取原始图片和标注图片
def read_voc_images(voc_dir, crop_size, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        # 过滤掉尺寸不足的图片
        feature_path=os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')
        label_path=os.path.join(voc_dir, 'SegmentationClass' ,f'{fname}.png')
        img=torchvision.io.read_image(feature_path)
        if img.shape[1] >= crop_size[0] and img.shape[2] >= crop_size[1]:
            features.append(feature_path)
            labels.append(label_path)
    return features, labels

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        # VOC数据集的3通道各自的均值和误差
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.features, self.labels = read_voc_images(voc_dir, crop_size, is_train=is_train)
        self.colormap2label = voc_colormap2label()
        #('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255) # 像素先转0~1，再标准化

    def __getitem__(self, idx):
        mode = torchvision.io.image.ImageReadMode.RGB

        # 读磁盘图片
        feature=torchvision.io.read_image(self.features[idx], mode)
        label=torchvision.io.read_image(self.labels[idx], mode)

        # 随机裁剪
        feature, label = voc_rand_crop(feature, label, *self.crop_size)

        # 标注图片的像素转分类
        return (self.normalize_image(feature), voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


if __name__ == '__main__':
    print('loading model...')
    #weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    #model = torch.hub.load("pytorch/vision", "fcn_resnet50", weights=weights)

    # 这个模型尺寸小，可以装进GPU，https://pytorch.org/vision/stable/models.html#table-of-all-available-semantic-segmentation-weights
    weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model=torch.hub.load('pytorch/vision','lraspp_mobilenet_v3_large',weights=weights) 
    model=model

    print('loading dataset...')
    dataset=VOCSegDataset(True,crop_size=(320, 480),voc_dir='./vocdataset/VOCdevkit/VOC2012/')

    # 多分类交叉熵,不需要自己做softmax
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

    # 开始训练
    print('starting train...')
    epoch=1000
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,num_workers=4) # 电脑内存不够，只能小batch了
    model.train()
    for i in range(epoch):
        batch_i=0
        for inputs,targets in dataloader:
            #print(inputs.shape,targets.shape)
            outputs=model(inputs)
            loss=loss_fn(outputs['out'],targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch={} batch={} loss={}'.format(i,batch_i,loss))
            batch_i+=1