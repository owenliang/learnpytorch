import torch
import matplotlib.pyplot as plt 
import numpy as np 

x=torch.tensor([5.0],requires_grad=True)
for i in range(100): # 100轮梯度下降，让y变小，但是x陷入了平台期，y无法继续减小，对于深度学习来说需要更多不同的输入数据产生不同形状的Loss函数，以便不断调整x
    y=torch.pow(x,2)+torch.pow(x,3)
    y.backward()

    # 调整x让y变小
    x.requires_grad_(False).add_(-0.01*x.grad).requires_grad_(True)
    # 清空本次梯度
    x.grad.zero_()
    # 看一下x梯度下降后,y是不是变小了
    with torch.no_grad():
        y=torch.pow(x,2)+torch.pow(x,3)
        print(x,y)

x=np.arange(-5,5)
y=np.power(x,2)+np.power(x,3)
plt.plot(x,y)
plt.show()