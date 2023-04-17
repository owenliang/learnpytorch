import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # X*W+B
            # x=(2,28*28)
            nn.Linear(28*28, 512),  # W=(28*28,512) ,B=(0.01,0.2,,,.......512个) ->
            nn.ReLU(), # 激活函数, activation，Relu(X)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

        self.bias=torch.rand(0) # Tensor,  (0.000145)


    def forward(self, x): # (1,28,28)
        # y=x^3+x^2+e^-x+0.000145
        y=torch.pow(x,3)+torch.pow(x,2)+torch.exp(-x)+self.bias
        return prob

model = NeuralNetwork()
model = model.to(device)

model = torch.compile(model)
model(x)

model=triton.compile(model,confi={'gpu':'gpu1,gpu2','node':'node1,node2'})
model(x)

l1=nn.Linear(28*28, 512),  # W=(28*28,512) ,B=(0.01,0.2,,,.......512个) ->
l2=nn.ReLU(), # 激活函数, activation，Relu(X)
l3=nn.Linear(512, 512),
l4=nn.ReLU(),
l5=nn.Linear(512, 10),
l6=nn.Softmax(dim=1)

import triton


@tf.function
def forward(x):
    x=l1(x)
    x=l2(x)
    x=l3(x)
    x=l4(x)
    x=l5(x)
    return x

X = torch.rand(15, 28, 28, device=device)
#print(X.shape,X.dtype)
prob = model(X)
'''
    input: (2,28,28)
        [
            [
                [12,35,53,3,5,6,6,...,]
                [12,35,53,3,5,6,6,...,]
                [12,35,53,3,5,6,6,...,]
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
            ],
            [
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
                12,35,53,3,5,6,6,...,
            ]
        ]
    flatten:
        [
            [ 12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,..., ],
            [ 12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,...,12,35,53,3,5,6,6,..., ]
        ]
    ...
        [ (2,512)
            [2,35,53,3,5,6,6,...,] ,
            [2,35,53,3,5,6,6,...,
        ]
    output:(2,10)
        [
            [0.1004, 0.0966, 0.0987, 0.0905, 0.1158, 0.0942, 0.0961, 0.1058, 0.1006,0.1013],
            [0.1004, 0.0966, 0.0987, 0.0905, 0.1158, 0.0942, 0.0961, 0.1058, 0.1006,0.813],
        ]
'''
print(prob)
y_pred = prob.argmax(1) # axis=1
'''
    [
        7,
        9,
    ]
'''
print(f"Predicted class: {y_pred}")

for name, param in model.named_parameters():
    print(name,param.requires_grad) # y=Tensor(3,requires_grad=False)*x+torch.rand().requires_grad(True)
    #print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")