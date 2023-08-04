import cv2
import numpy as np
import torch

IMG_WIDTH = 256
IMG_HEIGHT = 256
    
def read_image(image):
        
    image = np.array(image)
    width = image.shape[1]
    width_half = width // 2
    
    input_image = image[:, :width_half, :]
    target_image = image[:, width_half:, :]

    input_image = input_image.astype(np.float32)
    target_image = target_image.astype(np.float32)

    return input_image, target_image

def random_crop(image, dim):
    height, width, _ = dim
    rand_crop = np.random.uniform(low=0, high=256)
    x, y = np.random.uniform(low=0, high=int(height - 256)), np.random.uniform(low=0, high=int(width - 256))
    # return print(image.shape)
    return image[:, int(x):int(x) + 256, int(y):int(y) + 256]

def random_jittering_mirroring(input_image, target_image, height=286, width=286):
    
    # resizing to 286x286
    input_image = cv2.resize(input_image, (height, width))
    target_image = cv2.resize(target_image, (height, width))
    
    # print(input_image.shape)
    # cropping (random jittering) to 256x256
    stacked_image = np.stack([input_image, target_image], axis=0)
    cropped_image = random_crop(stacked_image, dim=[IMG_HEIGHT, IMG_WIDTH, 3])
    
    input_image, target_image = cropped_image[0], cropped_image[1]
    # print(input_image.shape)
    if torch.rand(()) > 0.5:
        # random mirroring
        input_image = np.fliplr(input_image)
        target_image = np.fliplr(target_image)
    return input_image, target_image
        
def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image
        
class Data_Normalize(object):
    def __init__(self, is_train=True) -> None:
        self.is_train = is_train
    
    def __call__(self, image):
        inp, tar = read_image(image)
        if self.is_train:
            inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy().transpose((2, 0, 1)))
        image_b = torch.from_numpy(tar.copy().transpose((2, 0, 1)))
        return image_a, image_b
