# %%
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# %%
if not os.path.isdir('train_data'):
    os.makedirs('train_data')
if not os.path.isdir('val_data'):
    os.makedirs('val_data')

# %%
n = 0
for file_name in os.listdir('trainA'):
    
    input_img = cv2.imread(f'trainA/{file_name}')
    target_img = cv2.imread(f'trainB/{file_name}')
    
    output_img = np.zeros((256, 512, 3))
    output_img[:, :256, :] = input_img
    output_img[:, 256:512, :] = target_img
    
    # print(input_img.shape, target_img.shape, output_img.shape)
    
    output_img
    if n % 10 == 0:
        cv2.imwrite(f'val_data/{file_name}', output_img)
    else:
        cv2.imwrite(f'train_data/{file_name}', output_img)
    
    n += 1
    
    # if n >= 20:
    #     break
    
# %%
