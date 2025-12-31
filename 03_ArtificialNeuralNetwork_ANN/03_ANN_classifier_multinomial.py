'''

'''

import torch

print(torch.__version__)
# 2.11.0.dev20251216+cu130

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# cuda

######################
## Data preparation ##
######################

'''
Firstly, install this library:
pip install ucimlrepo
'''

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X = covertype.data.features 
y = covertype.data.targets
'''Wait a little bit for it to fetch the data'''