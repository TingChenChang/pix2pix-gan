import os
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

TRAIN_DIR = 'data/edges2portrait/train_data'
VAL_DIR = 'data/edges2portrait/val_data/'
MODEL_DIR = 'training_weights/edges2portrait/'

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
n_gpus = 1
batch_size = 32 * n_gpus

# Dataloader
train_ds = ImageFolder(TRAIN_DIR, transform=transforms.Compose([Data_Normalize(is_train=True)]))
train_dl = DataLoader(train_ds, batch_size)

val_ds = ImageFolder(VAL_DIR, transform=transforms.Compose([Data_Normalize(is_train=False)]))
val_dl = DataLoader(val_ds, batch_size)

# def imshow(inputs, target, figsize=(10, 5)):
#     inputs = np.uint8(inputs)
#     target = np.uint8(target)
#     tar = np.rollaxis(target[0], 0, 3)
#     inp = np.rollaxis(inputs[0], 0, 3)
#     title = ['Input Image', 'Ground Truth']
#     display_list = [inp, tar]
#     plt.figure(figsize=figsize)
  
#     for i in range(2):
#         plt.subplot(1, 3, i + 1)
#         plt.title(title[i])
#         plt.axis('off')
#         plt.imshow(display_list[i])
#     plt.axis('off')
#     # plt.imshow(image)

# def show_batch(dl):
#     j = 0
#     for (images_a, images_b), _ in dl:
#         j += 1
#         imshow(images_a, images_b)
#         if j == 3:
#             break

# show_batch(train_dl)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # mps
torch.cuda.device_count()
print(f'Device: {device}')

norm_layer = get_norm_layer()

# Generator
generator = UnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False)
generator.apply(weights_init)
generator = torch.nn.DataParallel(generator)  # multi-GPUs
generator = generator.to(device)
print('# Generator Summary:')
print(summary(generator, (3, 256, 256)))

# Discriminator
discriminator = Discriminator(6, 64, n_layers=3, norm_layer=norm_layer)
discriminator.apply(weights_init)
discriminator = torch.nn.DataParallel(discriminator)  # multi-GPUs
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
LEARNING_RATE = 2e-4

G_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Training Loop
NUM_EPOCHS = 50

D_loss_plot, G_loss_plot = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    
    D_loss_list, G_loss_list = [], []
   
    for (input_img, target_img), _ in train_dl:
       
        D_optimizer.zero_grad()
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        
        # print("Inp:", input_img.shape)
        # print("Tar:", target_img.shape)
        
        generated_image = generator(input_img)
        # print("G_img:", generated_image.shape)
        
        real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
        
        # Train Discriminator with fake/real data
        # for fake data
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
        D_fake = discriminator(disc_inp_fake.detach())
        # print("D_fake:", D_fake.shape)
        D_fake_loss = discriminator_loss(D_fake, fake_target)
        # D_real_loss.backward()
    
        # for real data
        disc_inp_real = torch.cat((input_img, target_img), 1)
        D_real = discriminator(disc_inp_real)
        D_real_loss = discriminator_loss(D_real, real_target)
        # D_fake_loss.backward()
      
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)
      
        D_total_loss.backward()
        D_optimizer.step()
        
        # Train generator with real labels
        G_optimizer.zero_grad()
        fake_gen = torch.cat((input_img, generated_image), 1)
        G = discriminator(fake_gen)
        G_loss = generator_loss(generated_image, target_img, G, real_target)
        G_loss_list.append(G_loss)

        G_loss.backward()
        G_optimizer.step()
        
        print(f"D_total_loss: {D_total_loss:.6f}, G_loss:{G_loss:.6f}")
        
    print(f'Epoch: [{epoch}/{NUM_EPOCHS}]: D_loss: {torch.mean(torch.FloatTensor(D_loss_list)):.4f}, G_loss: {torch.mean(torch.FloatTensor(G_loss_list)):.4f}')
    print('#####')
    
    D_loss_plot.append(torch.mean(torch.FloatTensor(D_loss_list)))
    G_loss_plot.append(torch.mean(torch.FloatTensor(G_loss_list)))
    
    if epoch % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'discriminator_epoch_{epoch}.pth'))
        
        for (inputs, targets), _ in val_dl:
            inputs = inputs.to(device)
            generated_output = generator(inputs)
            save_image(generated_output.data[:10], os.path.join(MODEL_DIR, f'sample_{epoch}.png'), nrow=5, normalize=True)
