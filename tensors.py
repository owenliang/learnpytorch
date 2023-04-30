import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float32) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


shape = (3,28,28,)
rand_tensor = torch.rand(shape) # x*w+b
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape) # b

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


tensor = torch.rand(4, 4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)

tensor = torch.ones(3,4) 
print(tensor)   # (3,4)
print(tensor.T)  # (4,3)
print(tensor @ tensor.T) # (3,3)

y3 = torch.zeros(3,3)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)
#y1 = tensor @ tensor.T
#y2 = tensor.matmul(tensor.T)

tensor1=torch.tensor([1,2,3])
tensor2=torch.tensor([3,2,1])
z1 = tensor1 * tensor2
print(z1)

tensor1=torch.tensor([
   [ 1,2,3 ],
   [ 2,1,4 ],
])
tensor2=torch.pow(tensor1, 2) # element-wise 元素级
tensor3=torch.sqrt(tensor2) 
print(tensor3)


tensor=torch.tensor([
   [ 1,2,3 ],
   [ 2,1,4 ],
],dtype=torch.float32)
agg = tensor.sum()
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.div_(10)
print(tensor)

tensor = tensor.to("cuda")
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
n = t.numpy()
print(f"n: {n}")


n = np.ones(5)
t = torch.from_numpy(n)
n+=1
print(n)
print(t)