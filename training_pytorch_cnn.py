from pytorch_cnn import PytorchCNN
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 5 # number of epochs to train the model, the number of times the model sees the entire dataset
model = PytorchCNN(num_classes=2)
model = model.to(device) # move the model to the device
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # model.parameters() returns all the parameters of the model

train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 2, (100,))) # create random dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True) # DataLoader is used to load the data in batches

for epoch in range(num_epochs):
    model.train() # set the model to training mode
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device) # move the data to the device
        
        # forward pass
        logits = model(features) # logits are the raw predictions of the model
        loss = torch.nn.functional.cross_entropy(logits, targets) # cross-entropy loss is used for classification. It measures the difference between the predicted class probabilities and the actual class labels.

        # backward pass
        optimizer.zero_grad() # zero grad is used to clear the gradients of the model parameters because gradients are accumulated by default
        loss.backward() # compute the gradients of the loss with respect to the model parameters
        optimizer.step() # update the model parameters using the computed gradients