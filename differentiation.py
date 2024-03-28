import torch
from torch.autograd import grad

b = torch.tensor(0.1)
x1 = torch.tensor(0.2)
w1 = torch.tensor(0.3, required_grad=True)

u = w1 * x1
v = u + b
a = torch.sigmoid(v)
print(a)

# Compute the gradient of a with respect to w1
grad_a_wrt_w1 = grad(a, w1)