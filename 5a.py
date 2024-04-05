#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

# Set env CUDA_LAUNCH_BLOCKING=1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retina display
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

try:
    from einops import rearrange
except ImportError:
    get_ipython().run_line_magic('pip', 'install einops')
    from einops import rearrange


# In[18]:


import os

if os.path.exists('dog.jpg'):
    print('dog.jpg exists')
else:
    # Use curl on Windows or wget on Unix-like systems
    download_command = 'curl' if os.name == 'nt' else 'wget'
    os.system(f'{download_command} https://segment-anything.com/assets/gallery/AdobeStock_94274587_welsh_corgi_pembroke_CD.jpg -O dog.jpg')


# In[19]:


# Read in a image from torchvision
img = torchvision.io.read_image("dog.jpg")


# In[20]:


from sklearn import preprocessing

scaler_img = preprocessing.MinMaxScaler().fit(img.reshape(-1, 1))
scaler_img
img_scaled = scaler_img.transform(img.reshape(-1, 1)).reshape(img.shape)
img_scaled.shape

img_scaled = torch.tensor(img_scaled)
img_scaled = img_scaled.to(device)
img_scaled


# In[21]:


crop = torchvision.transforms.functional.crop(img_scaled.cpu(), 600, 800, 300, 300)
print(crop)
crop.shape
plt.imshow(rearrange(crop, 'c h w -> h w c').cpu().numpy())


# In[22]:


cropR = crop[0]
cropG = crop[1]
cropB = crop[2]
print(cropR)


# In[7]:


# Mask the image with NaN values 
def mask_image(img, prop):
    img_copy = img.clone()
    mask = torch.rand(img.shape) < prop
    img_copy[mask] = float('nan')
    return img_copy, mask
masked_img = mask_image(cropR, 0.01)
cropR = masked_img[0]
print(masked_img[1])
cropG[masked_img[1]] = float('nan')
cropB[masked_img[1]] = float('nan')
combined_tensor = torch.stack((cropR, cropG, cropB), dim=0)
combined_tensor = combined_tensor.detach().float()
print(combined_tensor)
plt.imshow(rearrange(combined_tensor, 'c h w -> h w c').cpu().numpy())


# In[24]:


import torch
import torch.nn as nn
import torch.optim as optim
def factorize(A, k, device=torch.device("cpu")):
    """Factorize the matrix D into A and B"""
    A = A.to(device)
    # Randomly initialize A and B
    W = torch.randn(A.shape[0], k, requires_grad=True, device=device)
    H = torch.randn(k, A.shape[1], requires_grad=True, device=device)
    # Optimizer
    optimizer = optim.Adam([W, H], lr=0.01)
    mask = ~torch.isnan(A)
    # Train the model
    for i in range(1000):
        # Compute the loss
        diff_matrix = torch.mm(W, H) - A
        diff_vector = diff_matrix[mask]
        loss = torch.norm(diff_vector)
        print(loss)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Backpropagate
        loss.backward()
        
        # Update the parameters
        optimizer.step()
       
    
    return W, H, loss
WR, HR, lossR = factorize(cropR, 150, device=device)
WG, HG, lossG = factorize(cropG, 150, device=device) 
WB, HB, lossB = factorize(cropB, 150, device=device)
RMSE = torch.mm(W, H) - crop


# In[25]:


R = torch.mm(WR,HR)
G = torch.mm(WG,HG)
B = torch.mm(WB,HB)
print(R)
print(G)
B


# In[26]:


combined_tensor = torch.stack((R, G, B), dim=0)
combined_tensor = combined_tensor.detach().float()
combined_tensor
combined_tensor.dtype


# In[27]:


plt.imshow(rearrange(combined_tensor, 'c h w -> h w c').cpu().numpy())


# In[23]:


import torch
import numpy as np

# Your existing cropR array
# cropR = torch.rand(300, 300).double()  # Assuming your data is in the range [0, 1]

# Set the random seed for reproducibility
# torch.manual_seed(42)

# Randomly choose the starting position of the 30x30 block
start_row = np.random.randint(0, 271)  # 300 - 30 + 1 = 271
start_col = np.random.randint(0, 271)

# Remove the 30x30 block by setting its values to NaN
cropR[start_row:start_row + 30, start_col:start_col + 30] = float('nan')
cropG[start_row:start_row + 30, start_col:start_col + 30] = float('nan')
cropB[start_row:start_row + 30, start_col:start_col + 30] = float('nan')

# Display the modified cropR array
print("Modified cropR array:")
print(cropR)
combined_tensor = torch.stack((cropR, cropG, cropB), dim=0)
combined_tensor = combined_tensor.detach().float()
print(combined_tensor)
plt.imshow(rearrange(combined_tensor, 'c h w -> h w c').cpu().numpy())


# In[ ]:
