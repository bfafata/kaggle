import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils



def train_model(model, criterion, optimizer, inputs, labels, writer, num_epochs=25):
    inputs = torch.Tensor(inputs).double()
    labels = torch.Tensor(labels).double()

    dataset = utils.data.TensorDataset(inputs, labels)
    loader = utils.data.DataLoader(dataset, batch_size=50)

    for epoch in range(num_epochs):
        losses=[]
        for i, (input, labels) in enumerate(loader):
            optimizer.zero_grad()

            outputs = model(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
            # Write to tensorboard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(loader) + i)
            if i%100==0:
                print(f"loss= {loss.item()}")

        print(f'Epoch {epoch+1}/{num_epochs} Loss: {np.mean(np.array(losses))}')

    writer.close()
    print('Finished Training')