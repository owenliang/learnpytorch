import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor() # (28,28) -> (28*28)
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=128,num_workers=8,pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=128,num_workers=8,pin_memory=True)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten() # (28,28) -> (28*28)  -> [0.2,0.5,0.3........ ] 784个输入x
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512), # 512个神经元 -> 512个输出,   y=w*x+b
                nn.ReLU(), # 激活函数 512个激活后的信号
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10), # 最后一层10个神经元
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x) # [64,10]
            return logits

    learning_rate = 1e-3 # 10^-3 -> 0.001
    batch_size = 64
    epochs = 5

    model = NeuralNetwork().to('cuda')
    # model=torch.compile(model)

    loss_fn = nn.CrossEntropyLoss() # Initialize the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X=X.to('cuda')
            y=y.to('cuda')
            pred = model(X) # pred=(64,10)
            loss = loss_fn(pred, y)  #  pred=[0.2,0.1,0.3,0.05,0.01,......,0.1],   y=[0,0,0,0,0,0,1,0,0,0,0,0]  -> ((0.2-0)^2+(0.1-0)&^+....(0.01-1)^2)/10 -> 0.7
            '''
                pred=[
                        [0.2,0.1,0.3,0.05,0.01,......,0.1], 每一行10个数字，表示每个分类的可能性   -> 0.75
                        [0.2,0.1,0.3,0.05,0.01,......,0.1], -> 0.3
                        [0.2,0.1,0.3,0.05,0.01,......,0.1], -> 0.3
                        ....
                        # 一共64行，表示每个样本各自的预测结果
                ]

            '''

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # for param in model.parameters():
            #     param.add_(-0.001*param.grad) # SDG
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset) # 10000
        num_batches = len(dataloader) # 10000/64
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader: # 10000张图片
                X=X.to('cuda')
                y=y.to('cuda')
                pred = model(X) # forward
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        # test_loop(test_dataloader, model, loss_fn)
    
    print(model)
    print("Done!")
