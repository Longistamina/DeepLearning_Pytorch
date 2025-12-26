'''
0. Prepare Data, Model, Loss, Optimizer

1. Integrate Validation and val_set into the Training loop.

2. Tune the model (explanation only)

3. Test the model with test_set (final exam)
'''

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


#----------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 0. Prepare Data, Model, Loss, Optimizer  -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

#################################
## Create X in ascending order ##
#################################

np.random.seed(24)
X = torch.tensor(
        np.random.uniform(low=1, high=11, size=(200, 1)),
        dtype=torch.float32,
        device=device
    ).sort(dim=0).values

torch.manual_seed(24)
X += torch.normal(mean=2.5, std=1, size=(200, 1)) # Add variation

#################################
## Create y in ascending order ##
#################################

np.random.seed(25)
y = torch.tensor(
        np.random.uniform(low=100, high=150, size=(200,)),
        dtype=torch.float32,
        device=device
    ).sort(dim=0).values

torch.manual_seed(25)
y += torch.normal(mean=10, std=1, size=(200,)) # Add variation

##########################
## Train-Val-Test split ##
##########################

train_len = int(0.7 * len(X)) # MUST be INTEGER
val_len = int(0.15 * len(X))
test_len = len(X) - (train_len + val_len)

print(train_len, val_len, test_len)
# 140 30 30

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X, y)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=16, shuffle=True)
val_set = DataLoader(val_split, batch_size=16, shuffle=True)
test_set = DataLoader(test_split, batch_size=16, shuffle=True)

#################
## Build model ##
#################

from torch import nn

class LinearRegressionModel(nn.Module):  
    def __init__(self):
        super().__init__()
        self.coefs = nn.Parameter(torch.randn(size=(1, ), requires_grad=True, dtype=torch.float32)) # initialize self.coefs as a random number
        self.bias = nn.Parameter(torch.randn(size=(1, ), requires_grad=True, dtype=torch.float32)) # initialize self.bias as a random number
        
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.coefs*X + self.bias
    
##################
## Define model ##
##################

torch.manual_seed(42)
model = LinearRegressionModel()
model.to(device)

##########
## Loss ##
##########

loss_fn = nn.MSELoss() # mean squared error

###############
## Optimizer ##
###############

optimizer = torch.optim.SGD(
    params=model.parameters(), # Parameters of the model that need to be optimized
    lr=2e-5,                   # The higher the learning rate, the more the parameters will be adjusted after every training step
)


#----------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------- 1. Integrate Validation and val_set into the Training loop. -------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

epochs = 10

for epoch in range(epochs):
    print("+"*50)
    # --- TRAINING ---
    _ = model.train() # Turn on training mode, enable gradient tracking
    for X_batch, y_batch in train_set:
        # (Standard training steps: forward, loss, zero_grad, backward, step)
        y_preds = model(X_batch).squeeze()
        loss = loss_fn(y_preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # --- VALIDATION (Every epoch) ---
    _ = model.eval() # 1. Set model to evaluation mode
    val_loss = 0
    with torch.inference_mode(): # 2. Turn off gradient tracking to save memory
        for X_val, y_val in val_set: # 3. Iterate through val_set
            val_preds = model(X_val).squeeze()
            # Accumulate loss to get an average for the whole set
            val_loss += loss_fn(val_preds, y_val).item()
    
    avg_val_loss = val_loss / len(val_set)
    
    print(f"Epoch: {epoch + 1}")
    print(f"Train loss: {loss:.4f}")
    print(f"Validation loss: {avg_val_loss:.4f}")
    
    '''
    ++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 1
Train loss: 10898.9717
Validation loss: 11243.5762
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 2
Train loss: 9937.9355
Validation loss: 10864.8428
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 3
Train loss: 10017.9365
Validation loss: 10488.9961
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 4
Train loss: 9939.7266
Validation loss: 10162.0581
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 5
Train loss: 9503.5547
Validation loss: 9812.7061
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 6
Train loss: 9308.4229
Validation loss: 9492.4849
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 7
Train loss: 9079.5312
Validation loss: 9184.7153
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 8
Train loss: 8713.5469
Validation loss: 8884.4946
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 9
Train loss: 8591.8389
Validation loss: 8601.7676
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 8297.2793
Validation loss: 8320.9907
'''


#----------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------- 2. Tune the model (explanation only) -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#
'''
Tuning or Fine-tuning typically means looking at your Validation Loss results and making decisions:

If Val Loss is much higher than Train Loss: Your model is "overfitting" (memorizing). 
=> You might need to simplify the model or add more data.

If both losses are high
=> You might need to increase the epochs or adjust the learning rate lr in your optimizer.

Adjustment: You change the parameters in your script (like lr=0.001) and run the whole training process again.
'''


#----------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- 3. Test the model with test_set (final exam) --------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

# --- FINAL TESTING (After all epochs are done) ---
_ = model.eval()

test_loss = 0
with torch.inference_mode():
    for X_test, y_test in test_set: # Use the test_set loader
        test_preds = model(X_test)
        test_loss += loss_fn(test_preds, y_test.unsqueeze(1)).item()

print(f"Final Test Loss: {test_loss / len(test_set):.4f}")
# Final Test Loss: 8306.1523