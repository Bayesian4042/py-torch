# What is Pytorch
1. It is a tensor library
2. It is an automatic differentiation engine
3. It is a deep learning library

# Tensor Library
1. tensor is an generalization of concepts like scalar (rank 0 tensor), vector and matrix.

```
scalar (rank-0 tensor)
import torch
a = torch.tensor(1.)
a.shape
------------
Vector (rank-1 tensor)
import torch
a = torch.tensor([1., 2., 3.])
a.shape
------------
Vector (rank-1 tensor)
import torch
a = torch.tensor([[1., 2., 3.],
                    [2., 3., 4.]])
a.shape
-------------

3D tensor
(rank-3 tensor) -> image (stacked matrices)

import torch
a = torch.tensor([[1., 2., 3.],
                    [2., 3., 4.]],
                    [[1., 2., 3.],
                    [2., 3., 4.]])
a.shape
------------


4D tensor
(rank-4 tensor) -> batches of image (stacked matrices)

import torch
a = torch.tensor([[1., 2., 3.],
                    [2., 3., 4.]],
                    [[1., 2., 3.],
                    [2., 3., 4.]],
                 [[1., 2., 3.],
                    [2., 3., 4.]],
                    [[1., 2., 3.],
                    [2., 3., 4.]]
                )
a.shape
```

# Automatic Differentiation Engine

```
Logistic Regression

[computation graph](computation-graph.jpeg)

    
```