import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
import matplotlib.pyplot as plt

# 6w
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

img,label=training_data[10]
print(label)


# img,label=training_data[5]
# print(img)

# 1w
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor() 
)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader)) # 64个样本，1各批次
# print(f"Feature batch shape: {train_features.size()}") # (64,1,28,28)
# print(f"Labels batch shape: {train_labels.size()}") # (64,)

# img = train_features[0].squeeze() # (28,28)
# label = train_labels[0] # 1
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

epochs = 10 # 10轮，每轮60000张
for i in range(epochs):
    for batch_idx,(batch_x,batch_y) in enumerate(train_dataloader):
        # model.train(batch_x,batch_y)  -> forward -> pred_y -> loss(pred_y,batch_y) -> backward() -> parameters grads -> optimize apply (parameters weights - learning_rate*parametes grads) 
        #print(batch_x,batch_y)
        print(f'epoch[{i}] batch[{batch_idx}]')



'''
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx] # (1,28,28), 误差交叉熵
    print(img)
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''