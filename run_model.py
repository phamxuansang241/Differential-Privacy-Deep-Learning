from DP_DL import data_lib, dp_lib, model_lib, get_args, train_valid_model
import json
import torch
from torch.utils.data import TensorDataset
from torch.optim import Adam
from torch import nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
    ARGUMENT PARSER AND UNPACK JSON OBJECT
"""
args = get_args()

with open(args.config_path, 'r') as openfile:
    config = json.load(openfile)

data_name = config['data_name']
epochs = config['epochs']
batch_size = config['batch_size']
lr = config['lr']
q = config['q']
epsilon = config['epsilon']
delta = config['delta']
clipping_norm = config['clipping_norm']

"""
    PREPROCESSING DATASET AND SETTING DATASET
"""
DataSetup = data_lib.DataSetup(data_name)
DataSetup.setup()

(x_train, y_train), (x_test, y_test) = DataSetup.get_data()

"""
    CREATING DATALOADER
"""
tensor_x_train = torch.from_numpy(x_train)
tensor_y_train = torch.from_numpy(y_train)

tensor_train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
data_size = len(tensor_train_dataset)

"""
    SETTING MODEL AND HYPERPARAMETERS FOR MODEL
"""
model_function = model_lib.get_model_function(data_name)
model = model_function()
model.to(device)
optimizer = Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss(reduction='none')


max_step = epochs
sigma = dp_lib.get_min_sigma(q, max_step, delta, epsilon)
print("First sigma: ", sigma)
sigma = 0.01
print("Sigma - Noise added to gradients", sigma)
train_valid_model.train(epochs, model, criterion, optimizer, tensor_train_dataset, 
                        batch_size, q, clipping_norm, sigma, device)




    