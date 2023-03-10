import torch
import numpy as np
from DP_DL import dp_lib
from torch.utils.data import TensorDataset, DataLoader
import time


"""
    TRAINING FUNCTION
"""

def train_model(epochs, model, criterion, optimizer, tensor_train_dataset,
          batch_size, q, clipping_norm, sigma, device
):


    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total_correct = 0
        # randomly select q fraction samples from data
        # according to the privacy analysis of moments accountant
        # training "Lots" are sampled by poisson sampling
        idx = np.where(np.random.rand(len(tensor_train_dataset[:][0])) < q)[0]

        data_size = len(tensor_train_dataset)
        sampled_dataset = TensorDataset(tensor_train_dataset[idx][0], tensor_train_dataset[idx][1])
        sample_data_loader = DataLoader(
            dataset=sampled_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        optimizer.zero_grad()

        clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        for batch_x, batch_y in sample_data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_y = model(batch_x.float())
            loss = criterion(pred_y, batch_y.long())
            
            total_loss = total_loss + loss.sum().item() / batch_size
            total_correct = total_correct + (pred_y.argmax(1) == batch_y).type(torch.float).sum().item()

            # bound l2 sensitivity (gradient clipping)
            # clip each of the gradient in the "Lot"
            for i in range(loss.size()[0]):
                loss[i].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipping_norm)
                for name, param in model.named_parameters():
                    clipped_grads[name] += param.grad 
                model.zero_grad()

            
        # add Gaussian noise
        for name, param in model.named_parameters():
            clipped_grads[name] += dp_lib.gaussian_noise(clipped_grads[name].shape, clipping_norm, sigma, device=device)
            
        # scale back
        for name, param in model.named_parameters():
            clipped_grads[name] /= (data_size*q)
        
        for name, param in model.named_parameters():
            param.grad = clipped_grads[name]
        
        # update local model
        optimizer.step()

        end_time = time.time()
        print("\t\t Total time taken to train: {:.2f}s".format(end_time - start_time))
        print(
            f"\t Train Epoch: {epoch} \t"
            f"Loss: {total_loss / len(sample_data_loader):.6f} \t"
            f"Acc: {total_correct / (data_size*q) * 100:.6f}"
        )


def test_model(model, criterion, tensor_test_dataset, batch_size, device):
    total_test_loss = 0
    test_correct = 0

    data_loader = DataLoader(
            dataset=tensor_test_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    with torch.no_grad():
        model.eval()

        for (x_batch, y_batch) in data_loader:
            (x_batch, y_batch) = (x_batch.to(device), y_batch.long().to(device))

            pred = model(x_batch)
            total_test_loss = total_test_loss + criterion(pred, y_batch)
            test_correct = test_correct + (pred.argmax(1) == y_batch).type(
                torch.float
            ).sum().item()

    avg_test_loss = total_test_loss / len(data_loader)
    test_correct = test_correct / len(tensor_test_dataset)

    results_dict = {
        'loss': avg_test_loss.cpu().detach().item(),
        'accuracy': test_correct
    }

    print("\t----- Evaluating on test dataset -----")
    print('{0}: {1}'.format('\tLoss', results_dict['loss']))
    print('{0}: {1}'.format('\tAccuracy', results_dict['accuracy']))
    print('-' * 100)


        