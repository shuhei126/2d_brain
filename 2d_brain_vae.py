"""
Train Soft-Intro VAE for image datasets
Author: Tal Daniel
"""

import argparse
# standard
import os
import pickle
import random
import time
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
# imports
# torch and friends
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import os.path as osp
import torchio as tio
import torchvision.utils as vutils
from sklearn.model_selection import GroupShuffleSplit, train_test_split, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchio.transforms.augmentation.intensity.random_bias_field import \
    RandomBiasField
from torchio.transforms.augmentation.intensity.random_noise import RandomNoise
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
from torchvision.utils import make_grid
from tqdm import tqdm

import utils.my_trainer as trainer
import utils.train_result as train_result
from dataset import DigitalMonstersDataset, ImageDatasetFromFile
from datasets.dataset import CLASS_MAP, load_data
from metrics.fid_score import calculate_fid_given_dataset
from utils.data_class import BrainDataset

matplotlib.use('Agg')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True #この行をFalseにすると再現性はとれるが、速度が落ちる
    torch.backends.cudnn.deterministic = True
    return
fix_seed(0)

CLASS_MAP = {
    "CN": 0,
    "AD": 1,
    "EMCI": 2,
    "LMCI": 3,
    "MCI": 4,
    "SMC": 5,
}
SEED_VALUE = 0

data = load_data(kinds=["ADNI2","ADNI2-2"], classes=["CN","AD","EMCI","LMCI","MCI","SMC"],size="half",unique=False, blacklist=False)

pids = []
voxels = np.zeros((len(data), 80, 96, 80))
labels = np.zeros(len(data))
for i in tqdm(range(len(data))):
    pids.append(data[i]["pid"])
    voxels[i] = data[i]["voxel"]
    # voxels[i] = normalize(voxels[i], np.min(voxels[i]), np.max(voxels[i]))
    labels[i] = CLASS_MAP[data[i]["label"]]
pids = np.array(pids)

gss = StratifiedGroupKFold(n_splits = 5,shuffle= True, random_state=42)
tid, vid = list(gss.split(voxels,labels, groups=pids))[0]
train_voxels = voxels[tid]
val_voxels = voxels[vid]
train_labels = labels[tid]
val_labels = labels[vid]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class BuildingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bias=False):
        super(BuildingBlock, self).__init__()
        self.res = stride == 1
        self.shortcut = self._shortcut()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=stride),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_ch),
        )

    def _shortcut(self):
        return lambda x: x

    def forward(self, x):
        if self.res:
            shortcut = self.shortcut(x)
            return self.relu(self.block(x) + shortcut)
        else:
            return self.relu(self.block(x))

class UpsampleBuildingkBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bias=False):
        super(UpsampleBuildingkBlock, self).__init__()
        self.res = stride == 1
        self.shortcut = self._shortcut()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=stride),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_ch),
        )

    def _shortcut(self):
        return lambda x: x

    def forward(self, x):
        if self.res:
            shortcut = self.shortcut(x)
            return self.relu(self.block(x) + shortcut)
        else:
            return self.relu(self.block(x))

class ResNetEncoder(nn.Module):
    def __init__(self, in_ch, block_setting):
        super(ResNetEncoder, self).__init__()
        self.block_setting = block_setting
        self.in_ch = in_ch
        last = 1
        blocks = [nn.Sequential(
            nn.Conv2d(1, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )]
        for line in self.block_setting:
            c, n, s = line[0], line[1], line[2]
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(nn.Sequential(BuildingBlock(in_ch, c, stride)))
                in_ch = c
        self.inner_ch = in_ch
        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, last, kernel_size=1, stride=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        return self.conv(h)

class ResNetDecoder(nn.Module):
    def __init__(self, encoder: ResNetEncoder, blocks=None):
        super(ResNetDecoder, self).__init__()
        last = encoder.block_setting[-1][0]
        if blocks is None:
            blocks = [nn.Sequential(
                nn.Conv2d(1, last, 1, 1, bias=True),
                nn.BatchNorm2d(last),
                nn.ReLU(inplace=True),
            )]
        in_ch = last
        for i in range(len(encoder.block_setting)):
            if i == len(encoder.block_setting) - 1:
                nc = encoder.in_ch
            else:
                nc = encoder.block_setting[::-1][i + 1][0]
            c, n, s = encoder.block_setting[::-1][i]
            for j in range(n):
                stride = s if j == n-1 else 1
                c = nc if j == n-1 else c
                blocks.append(nn.Sequential(UpsampleBuildingkBlock(in_ch, c, stride)))
                in_ch = c
        blocks.append(nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

class BaseEncoder(nn.Module):
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()
class BaseDecoder(nn.Module):
    def __init__(self) -> None:
        super(BaseDecoder, self).__init__()

class BaseCAE(nn.Module):
    def __init__(self) -> None:
        super(BaseCAE, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
    def encode(self, x):
        z = self.encoder(x)
        return z
    def decode(self, z):
        out = self.decoder(z)
        return out
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    def decode(self, vec):
        out = self.decoder(vec)
        return out
    def reparameterize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encode(x)
        vec = self.reparameterize(mu, logvar)
        x_hat = self.decode(vec)
        return x_hat, vec, mu, logvar

class ResNetCAE(BaseCAE):
    def __init__(self, in_ch, block_setting) -> None:
        super(ResNetCAE, self).__init__()
        self.encoder = ResNetEncoder(
            in_ch=in_ch,
            block_setting=block_setting,
        )
        self.decoder = ResNetDecoder(self.encoder)

class VAEResNetEncoder(ResNetEncoder):
    def __init__(self, in_ch, block_setting) -> None:
        super(VAEResNetEncoder, self).__init__(in_ch, block_setting)
        self.mu = nn.Conv2d(self.inner_ch, 1, kernel_size=1, stride=1, bias=True)
        self.var = nn.Conv2d(self.inner_ch, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor):
        h = self.blocks(x)
        mu = self.mu(h)
        var = self.var(h)
        return mu, var
        
class ResNetVAE(BaseVAE):
    def __init__(self, in_ch, block_setting) -> None:
        super(ResNetVAE, self).__init__()
        self.encoder = VAEResNetEncoder(
            in_ch=in_ch,
            block_setting=block_setting,
        )
        self.decoder = ResNetDecoder(self.encoder)

def mse_loss(out, x):
    bsize = x.size(0)
    x = x.view(bsize, -1)
    out = out.view(bsize, -1)
    mse = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    return mse

def kld_loss(mu, logvar):
    bsize = mu.size(0)
    mu = mu.view(bsize, -1)
    logvar = logvar.view(bsize, -1)
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

def normal_loss(x_hat, mu, logvar, x, msew=1, kldw=10):
    mse = mse_loss(x_hat, x) * msew
    kld = kld_loss(mu, logvar) * kldw
    loss = mse + kld
    return loss, mse, kld

def str_to_list(x):
    return [int(xi) for xi in x.split(',')]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_image(writer, image_list, cur_iter, num_rows=8):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=num_rows), cur_iter)


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" +"brain_256_0.2_0.2_32_2/"+ prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/brain_256_0.2_0.2_32_2/"):
        os.makedirs("./saves/brain_256_0.2_0.2_32_2/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))

def train_vae(dataset='brain', lr=1e-3, batch_size=1, num_workers=28,start_epoch=0, exit_on_negative_diff=False,
                         num_epochs=1, save_interval=50, recon_loss_type="mse",
                         test_iter=1000, seed=-1, pretrained=None, device=torch.device("cpu"), 
                         num_row=8, with_fid=False, train_voxels=train_voxels, train_labels=train_labels, val_voxels=val_voxels, val_labels=val_voxels):
    """
    :param dataset: dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', 'celeb256', 'celeb1024']
    :param z_dim: latent dimensions
    :param lr_e: learning rate for encoder
    :param lr_d: learning rate for decoder
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param num_vae: number of epochs for vanilla vae training
    :param save_interval: epochs between checkpoint saving
    :param recon_loss_type: type of reconstruction loss ('mse', 'l1', 'bce')
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param num_row: number of images in a row gor the sample image saving
    :param gamma_r: coefficient for the reconstruction loss for fake data in the decoder
    :param with_fid: calculate FID during training (True/False)
    :return:
    """
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # --------------build models -------------------------
    if dataset == 'cifar10':
        image_size = 32
        channels = [64, 128, 256]
        train_set = CIFAR10(root='./cifar10_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 3
    elif dataset == 'celeb128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        output_height = 128
        train_size = 162770
        data_root = '../data/celeb256/img_align_celeba'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
        output_height = 256
        # train_size = 162770
        train_size = 4000
        data_root = '/data2/tomoshi/celeba_hq_256'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
        print(len(train_set))
    elif dataset == 'celeb1024':
        channels = [16, 32, 64, 128, 256, 512, 512, 512]
        image_size = 1024
        ch = 3
        output_height = 1024
        train_size = 29000
        data_root = './' + dataset
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0

        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'monsters128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        data_root = './monsters_ds/'
        train_set = DigitalMonstersDataset(root_path=data_root, output_height=image_size)
    elif dataset == 'svhn':
        image_size = 32
        channels = [64, 128, 256]
        train_set = SVHN(root='./svhn', split='train', transform=transforms.ToTensor(), download=True)
        ch = 3
    elif dataset == 'fmnist':
        image_size = 28
        channels = [64, 128]
        train_set = FashionMNIST(root='./fmnist_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 1
    elif dataset == 'mnist':
        image_size = 28
        channels = [64, 128]
        train_set = MNIST(root='./mnist_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 1
    elif dataset == 'brain':
        image_size = 80
        # channels = [64, 128, 256, 256, 512, 512]
        # train_set = BrainDataset(train_voxels, train_labels,transforms.Compose(train_transforms_list))
        # val_set = BrainDataset(val_voxels, val_labels,transforms.Compose(val_transforms_list))
        train_set = BrainDataset(train_voxels, train_labels)
        val_set = BrainDataset(val_voxels, val_labels)
        ch = 1
    else:
        raise NotImplementedError("dataset is not supported")

    model = ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]]).to(device)
    # model = ResNetVAE(16, [[16,1,2],[32,1,2],[64,2,2],[128,2,2]], conditional=False).to(device)
    # model = ResNetVAE(32, [[32,1,2],[64,1,2],[64,2,2],[128,2,2]], conditional=False).to(device)

    log_path = "./logs/" + "output" + "_vae/"
    
    def init_weights_he(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        return

    model.apply(init_weights_he)
    if pretrained is not None:
        load_model(model, pretrained, device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
    print(len(train_dataloader))
    start_time = time.time()

    train_loss_list, val_loss_list =[], []
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_dataloader:
            # train_inputs = torch.zeros((96,1,80,80))
            # for i in range(96):
            #     train_inputs[i] = inputs[:,:,:,i,:]
            train_inputs = torch.zeros((80,1,80,80))
            for i in range(80):
                train_inputs[i] = inputs[:,:,:,i+8,:]
            train_inputs = train_inputs.to(device)
            optimizer.zero_grad()
            x_re,_,mu,logvar = model.forward(train_inputs)
            loss,_,_ = normal_loss(x_re, mu, logvar, train_inputs,1,1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(len(train_dataloader.dataset))
        train_loss /= (len(train_dataloader.dataset)*96)
        train_loss_list.append(train_loss)
    
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                val_inputs = torch.zeros((96,1,80,80))
                for i in range(96):
                    val_inputs[i] = inputs[:,:,:,i,:]
                val_inputs = val_inputs.to(device)
                inputs = inputs.to(device)
                x_re,_, mu, logvar = model.forward(val_inputs)
                #labels = labels.to(device)
                loss,_,_ = normal_loss(x_re, mu, logvar, val_inputs,1,1)

                val_loss += loss.item()
                
            val_loss /= (len(val_dataloader.dataset)*96)
            val_loss_list.append(val_loss)
        elapsed_time = time.time()
        print("Epoch [%3d], train_loss: %f, val_loss: %f, elapsed time %d秒"
                % (
                    epoch + 1,
                    train_loss,
                    val_loss,
                    elapsed_time - start_time,
                )
        )
    
    torch.save(model.state_dict(), log_path + "vae_weight_1_2_80.pth")
    os.makedirs(log_path + "/img", exist_ok=True)
    epoch = num_epochs
    plt.rcParams["font.size"] = 18
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.plot(range(1, epoch + 1), train_loss_list, label="train_loss")
    ax1.plot(range(1, epoch + 1), val_loss_list, label="val_loss")
    ax1.set_title("loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    fig1.savefig(log_path + "/img/loss_1_1_80.png")



if __name__ == '__main__':
    """
    Recommended hyper-parameters:
    - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
    - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """


    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(device)
    try:
        train_vae(dataset="brain", batch_size=1, num_workers=4, num_epochs=300,
                             device=device, save_interval=50, start_epoch=0, 
                             pretrained=None,test_iter=1920, with_fid=False)
    except SystemError:
        print("Error, probably loss is NaN, try again...")