# yolo推理
import torch

model = torch.hub.load('./yolov5-master', 'custom', path='yolov5-master/runs/train/exp4/weights/best.pt', source='local') 

# Image
img = 'datasets/coco128/images/train2017/000000000034.jpg'
# Inference
results = model(img)
#print(type(results))
# Results, change the flowing to: results.show()
# results.show()  # or .show(), .save(), .crop(), .pandas(), etc
df=results.pandas()
print(df.xyxy[0]) # 取第0张图片的预测结果(N个框)
results.crop() # 截取目标
print(df.xyxy[0].to_json(orient="records"))  # JSON img1 predictions