import torch 
import matplotlib.pyplot as plt 
import numpy as np 

# loss=w^2

# 随机初始化
w=torch.tensor([99.995,],requires_grad=True)

# 训练
loss_list=[]
for i in range(10000):
    # 前向传播
    loss=torch.pow(w,2)
    print('w=',w,'loss=',loss)
    loss_list.append(loss.detach().numpy())
    # 反向传播
    loss.backward()
    # 打印一下
    #print(w)
    #print(w.grad)
    # 梯度下降
    w.requires_grad_(False).add_(-0.001*w.grad).requires_grad_(True)
    # w-=0.001*w.grad # 99.9950-0.001*199.99  # optimizer
    # 清理本轮backward记录在w里面的grad
    w.grad.zero_()

plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()

x=np.arange(-200,200,1) # [-200,-199,.,0,..,199,200]
y=np.power(x,2)
plt.plot(x,y)

plt.xlabel('w')
plt.ylabel('loss')
plt.title('loss=w^2')
plt.show()

'''

def loss(w):
    return torch.pow(w,2) # loss=w^2

# 模型初始的Loss很大，需要梯度下降让loss变小，如何让Loss变小？ 调整w
print(loss(w))

points_w=[]
points_loss=[]
for i in range(1000):
    w1=w+1
    w2=w-1
    if loss(w1)>loss(w2):
        w=w2 
    else:
        w=w1
    points_w.append(w)
    points_loss.append(loss(w))
    print('w=',w,'loss=',loss(w))


x=np.arange(-200,200,1) # [-200,-199,.,0,..,199,200]
y=np.power(x,2)
plt.plot(x,y)

mask=np.random.choice(np.arange(len(x)),10)
print(mask)

points_w=np.array(points_w)
points_loss=np.array(points_loss)
plt.scatter(points_w[mask], points_loss[mask])

plt.xlabel('w')
plt.ylabel('loss')
plt.title('loss=w^2')
plt.show()
'''