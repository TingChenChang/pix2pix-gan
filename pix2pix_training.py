# %%
import os
import numpy as np
import torch
import torch.nn as nn
import functools
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from data_preprocess import Data_Normalize
from unet_generator import UnetGenerator
from patchgan_discriminator import Discriminator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,5,6,7'

TRAIN_DIR = 'data/edges2portrait/train_data/'
VAL_DIR = 'data/edges2portrait/val_data/'
MODEL_DIR = 'training_weights/edges2portrait/'

PRETRAINED_GENERATOR = ''
PRETRAINED_DISCRIMINATOR = ''

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # mps
n_gpus = max(torch.cuda.device_count(), 1)
print(f'Device: {device}, GPUs: {n_gpus}')
batch_size = 32 * n_gpus

if device == 'cuda':
    torch.cuda.empty_cache()

# Dataloader
train_ds = ImageFolder(TRAIN_DIR, transform=transforms.Compose([Data_Normalize(is_train=True)]))
train_dl = DataLoader(train_ds, batch_size)

val_ds = ImageFolder(VAL_DIR, transform=transforms.Compose([Data_Normalize(is_train=False)]))
val_dl = DataLoader(val_ds, batch_size)

# %%
# custom weights initialization called on generator and discriminator
def weights_init(net, init_type='normal', scaling=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_norm_layer():
    """Return a normalization layer
       For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    """
    norm_type = 'batch'
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer

norm_layer = get_norm_layer()

# Generator
generator = UnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False)
if PRETRAINED_GENERATOR:
    generator.load_state_dict(torch.load(PRETRAINED_GENERATOR))
else:
    generator.apply(weights_init)
# generator = torch.nn.DataParallel(generator)  # multi-GPUs
generator = generator.to(device)
print('# Generator Summary:')
print(summary(generator, (3, 256, 256)))

# Discriminator
discriminator = Discriminator(6, 64, n_layers=3, norm_layer=norm_layer)
if PRETRAINED_DISCRIMINATOR:
    discriminator.load_state_dict(torch.load(PRETRAINED_DISCRIMINATOR))
else:
    discriminator.apply(weights_init)
# discriminator = torch.nn.DataParallel(discriminator)  # multi-GPUs
discriminator = discriminator.to(device)
print('# Discriminator Summary:')
print(summary(discriminator, (6, 256, 256)))

# Loss Function
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    # print(gen_loss)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

# Optimizer
LEARNING_RATE = 1e-6

G_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Training Loop
NUM_EPOCHS = 100

D_loss_plot, G_loss_plot = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    
    D_loss_list, G_loss_list = [], []
    
    for (input_img, target_img), _ in train_dl:
       
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        
        # ground truth labels real and fake
        real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
        
        # generator forward pass
        generated_image = generator(input_img)
        
        # train discriminator with fake/generated images
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
        D_fake = discriminator(disc_inp_fake.detach())
        D_fake_loss = discriminator_loss(D_fake, fake_target)
    
        # train discriminator with real images
        disc_inp_real = torch.cat((input_img, target_img), 1)
        D_real = discriminator(disc_inp_real)
        D_real_loss = discriminator_loss(D_real, real_target)

        # average discriminator loss
        D_total_loss = D_real_loss + D_fake_loss
        D_loss_list.append(D_total_loss)
        # compute gradients and run optimizer step
        D_optimizer.zero_grad()
        D_total_loss.backward()
        D_optimizer.step()
        
        # train generator with real labels
        fake_gen = torch.cat((input_img, generated_image), 1)
        G = discriminator(fake_gen)
        G_loss = generator_loss(generated_image, target_img, G, real_target)
        G_loss_list.append(G_loss)
        # compute gradients and run optimizer step
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
        # print(f"D_total_loss: {D_total_loss:.6f}, G_loss:{G_loss:.6f}")
        
    print(f'Epoch: [{epoch}/{NUM_EPOCHS}]: D_loss: {torch.mean(torch.FloatTensor(D_loss_list)):.4f}, G_loss: {torch.mean(torch.FloatTensor(G_loss_list)):.4f}')
    
    D_loss_plot.append(torch.mean(torch.FloatTensor(D_loss_list)))
    G_loss_plot.append(torch.mean(torch.FloatTensor(G_loss_list)))
    
    if epoch % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'discriminator_epoch_{epoch}.pth'))
        
        for (inputs, targets), _ in val_dl:
            inputs = inputs.to(device)
            generated_output = generator(inputs)
            save_image(generated_output.data[:10], os.path.join(MODEL_DIR, f'sample_{epoch}.png'), nrow=5, normalize=True)

# %%
