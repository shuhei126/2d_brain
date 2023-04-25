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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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

"""
Models
"""
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

data = load_data(kinds=["ADNI2","ADNI2-2"], classes=["CN","AD","MCI","EMCI","LMCI","SMC"],size="half",unique=False, blacklist=False)
# classes=["CN","AD","MCI","EMCI","LMCI","SMC"]

pids = []
voxels = np.zeros((len(data), 80, 96, 80))
labels = np.zeros(len(data))
for i in tqdm(range(len(data))):
    pids.append(data[i]["pid"])
    voxels[i] = data[i]["voxel"]
    # voxels[i] = normalize(voxels[i], np.min(voxels[i]), np.max(voxels[i]))
    labels[i] = CLASS_MAP[data[i]["label"]]
pids = np.array(pids)

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
tid, vid = list(gss.split(voxels, groups=pids))[0]
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
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2,inplace=True),
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
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2,inplace=True),
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
            nn.LeakyReLU(0.2,inplace=True),
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
                nn.LeakyReLU(0.2,inplace=True),
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

class SoftIntroVAE(nn.Module):
    def __init__(self, in_ch, block_setting, conditional=False):
        super(SoftIntroVAE, self).__init__()
        self.conditional = conditional
        self.encoder = VAEResNetEncoder(in_ch=in_ch, block_setting=block_setting)
        self.decoder = ResNetDecoder(self.encoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_re = self.decoder(z)
        return mu, logvar, z, x_re

    def sample(self, z, y_cond=None):
        # x.view(-1, 2)
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, 1, 5, 5).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z, y_cond=None):
        y = self.decoder(z)
        return y
"""
Helpers
"""


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """

    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


# def reparameterize(mu, logvar):
#     """
#     This function applies the reparameterization trick:
#     z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
#     :param mu: mean of x
#     :param logvar: log variaance of x
#     :return z: the sampled latent variable
#     """
#     device = mu.device
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn_like(std).to(device)
#     return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error

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
    model_out_path = "./saves/" +"brain_11/"+ prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/brain_11/"):
        os.makedirs("./saves/brain_11/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


"""
Train Functions
"""


def train_soft_intro_vae(dataset='brain',  lr_e=2e-4, lr_d=2e-4, batch_size=8, num_workers=28,
                         start_epoch=0, exit_on_negative_diff=False,
                         num_epochs=1, num_vae=0, save_interval=50, recon_loss_type="mse",
                         beta_kl=0.5, beta_rec=1.0, beta_neg=1024, test_iter=1000, seed=-1, pretrained=None,
                         device=torch.device("cpu"), num_row=8, gamma_r=1e-8, with_fid=False, train_voxels=train_voxels, train_labels=train_labels, val_voxels=val_voxels, val_labels=val_voxels):
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

    # model = SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]], conditional=False).to(device)
    # model = SoftIntroVAE(16, [[16,1,2],[32,1,2],[64,2,2],[128,2,2]], conditional=False).to(device)
    model = SoftIntroVAE(32, [[32,1,2],[64,1,2],[64,2,2],[128,2,2]], conditional=False).to(device)
    # model = SoftIntroVAE(64, [[64,1,2],[128,1,2],[256,1,2],[512,2,2]], conditional=False).to(device)

    def init_weights_he(m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        return

    model.apply(init_weights_he)
    if pretrained is not None:
        load_model(model, pretrained, device)
    print(model)

    fig_dir = './figures_brain_11' + dataset
    os.makedirs(fig_dir, exist_ok=True)

    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    scale = 1 / (ch * image_size ** 2)  # normalize by images size (channels * height * width)

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
    print(len(train_data_loader))
    start_time = time.time()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    exp_elbos_f = []
    exp_elbos_r = []
    best_fid = None
    for epoch in range(start_epoch, num_epochs):
        if with_fid and ((epoch == 0) or (epoch >= 100 and epoch % 20 == 0) or epoch == num_epochs - 1):
            with torch.no_grad():
                print("calculating fid...")
                fid = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048,
                                                  device=device, num_images=50000)
                print("fid:", fid)
                if best_fid is None:
                    best_fid = fid
                elif best_fid > fid:
                    print("best fid updated: {} -> {}".format(best_fid, fid))
                    best_fid = fid
                    # save
                    save_epoch = epoch
                    prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                        beta_rec) + "_" + "fid_" + str(fid) + "_"
                    save_checkpoint(model, save_epoch, cur_iter, prefix)

        diff_kls = []
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()

        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []
        batch_exp_elbo_f = []
        batch_exp_elbo_r = []

        pbar = tqdm(iterable=train_data_loader)

        for batch in pbar:
            batch,_ = batch
            # --------------train------------
            if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
                batch = batch[0]
            if epoch < num_vae:
                if len(batch.size()) == 3:
                    if len(batch.size()) == 3:
                        batch = batch.unsqueeze(0)

                batch_size = batch_size

                real_batch = batch.to(device)

                # =========== Update E, D ================

                real_mu, real_logvar, z, rec = model(real_batch)

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                loss = beta_rec * loss_rec + beta_kl * loss_kl

                optimizer_d.zero_grad()
                optimizer_e.zero_grad()
                loss.backward()
                optimizer_e.step()
                optimizer_d.step()

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_kl.data.cpu().item())

                if cur_iter % test_iter == 0:
                    vutils.save_image(torch.cat([real_batch, rec], dim=0).data.cpu(),
                                      '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            else:
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                b_size = batch_size
                noise_batch = torch.randn(size=(b_size, 1, 5, 5)).to(device)

                real_batch = torch.zeros((len(batch)*96,1,80,80))
                for i in range(len(batch)):
                    for j in range(96):
                        real_batch[i*96+j] = batch[i,:,:,j,:]
                real_batch = real_batch.to(device)
                # real_batch = torch.zeros((len(batch)*80,1,80,80))
                # for i in range(len(batch)):
                #     for j in range(80):
                #         real_batch[i*80+j] = batch[i,:,:,j+8,:]
                # real_batch = real_batch.to(device)
                # =========== Update E ================
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False

                fake = model.sample(noise_batch)

                real_mu, real_logvar = model.encode(real_batch)
                z = model.reparameterize(real_mu, real_logvar)
                rec = model.decoder(z)
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                # loss_rec = mse_loss(rec,real_batch)

                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                # lossE_real_kl = kld_loss(real_mu, real_logvar)

                rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
                fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

                kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")########
                kl_fake = calc_kl(fake_logvar, fake_mu, reduce="sum")#sum or none ######
                # kl_rec = kld_loss(rec_mu, rec_logvar)
                # kl_fake = kld_loss(fake_mu,fake_logvar)

                loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='sum')#sum or none###
                # loss_rec_rec_e = mse_loss(rec_rec,rec)
 
                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                    print("test")
                loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
                # loss_rec_fake_e = mse_loss(rec_fake,fake)
                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)
                    print("test")

                expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake)).exp().mean()

                lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                # lossE_fake = 0.50 * (expelbo_rec + expelbo_fake)
                lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

                lossE = lossE_real + lossE_fake
                optimizer_e.zero_grad()
                lossE.backward()
                optimizer_e.step()

                # ========= Update D ==================
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                fake = model.sample(noise_batch)
                rec = model.decoder(z.detach())
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                # loss_rec = mse_loss(rec,real_batch)

                rec_mu, rec_logvar = model.encode(rec)
                z_rec = model.reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar = model.encode(fake)
                z_fake = model.reparameterize(fake_mu, fake_logvar)

                rec_rec = model.decode(z_rec.detach())
                rec_fake = model.decode(z_fake.detach())

                loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type,
                                                        reduction="mean")
                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
                                                         reduction="mean")
                # loss_rec_rec = mse_loss(rec_rec,rec.detach())
                # loss_fake_rec = mse_loss(rec_fake,fake.detach())

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
                # lossD_rec_kl = kld_loss(rec_mu,rec_logvar)
                # lossD_fake_kl = kld_loss(fake_mu,fake_logvar)

                lossD = scale * (loss_rec * beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                optimizer_d.zero_grad()
                lossD.backward()
                optimizer_d.step()
                if torch.isnan(lossD) or torch.isnan(lossE):
                    raise SystemError

                dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=lossE_real_kl.data.cpu().item(),
                                 diff_kl=dif_kl.item(), expelbo_f=expelbo_fake.cpu().item())

                diff_kls.append(-lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item())
                batch_kls_real.append(lossE_real_kl.data.cpu().item())
                batch_kls_fake.append(lossD_fake_kl.cpu().item())
                batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
                batch_rec_errs.append(loss_rec.data.cpu().item())
                batch_exp_elbo_f.append(expelbo_fake.data.cpu())
                batch_exp_elbo_r.append(expelbo_rec.data.cpu())

                if cur_iter % test_iter == 0:
                    # _, _, _, rec_det = model(real_batch, deterministic=True)
                    _, _, _, rec_det = model(real_batch)
                    max_imgs = min(batch_size*96, 16)
                    vutils.save_image(
                        torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                        '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            cur_iter += 1
        e_scheduler.step()
        d_scheduler.step()
        pbar.close()
        if exit_on_negative_diff and epoch > 50 and np.mean(diff_kls) < -1.0:
            print(
                f'the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)')
            print("try to lower beta_neg hyperparameter")
            print("exiting...")
            raise SystemError("Negative KL Difference")

        if epoch > 10:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))
            exp_elbos_f.append(np.mean(batch_exp_elbo_f))
            exp_elbos_r.append(np.mean(batch_exp_elbo_r))
            # epoch summary
            print('#' * 50)
            print(f'Epoch {epoch} Summary:')
            print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
            print(
                f'rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}')
            print(
                f'diff_kl: {np.mean(diff_kls):.3f}, exp_elbo_f: {exp_elbos_f[-1]:.4e}, exp_elbo_r: {exp_elbos_r[-1]:.4e}')
            print(f'time: {time.time() - start_time}')
            print('#' * 50)
        if epoch == num_epochs - 1:
            with torch.no_grad():
                # _, _, _, rec_det = model(real_batch, deterministic=True)
                _, _, _, rec_det = model(real_batch)
                noise_batch = torch.randn(size=(b_size, 1, 5, 5)).to(device)
                fake = model.sample(noise_batch)
                max_imgs = min(batch_size*96, 16)
                # vutils.save_image(
                #     torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                #     '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            # plot graphs
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
            ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
            ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
            ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
            ax.legend()
            plt.savefig('./brain_11.jpg')
            with open('./brain_11', 'wb') as fp:
                graph_dict = {"kl_real": kls_real, "kl_fake": kls_fake, "kl_rec": kls_rec, "rec_err": rec_errs}
                pickle.dump(graph_dict, fp)
            # save models
            prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, epoch, cur_iter, prefix)
            model.train()


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
#brain_00 beta_kl: 0.2, beta_rec: 0.4, beta_neg: 25,  batch_size: 8 ,1/alpha:0.25, chanel: 32 /loss change
#brain_0  beta_kl: 0.2, beta_rec: 0.4, beta_neg: 25,  batch_size: 8 ,1/alpha:0.25, chanel: 32 /good
#brain_1  beta_kl: 0.2, beta_rec: 0.4, beta_neg: 128, batch_size: 8 ,1/alpha:0.25, chanel: 32
#brain_2  beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, batch_size: 8 ,1/alpha:0.5,  chanel: 32
#brain_3  beta_kl: 0.2, beta_rec: 0.4, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 32
#brain_4  beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 12
#brain_5  beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 16
#brain_6  beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 32 /good
#brain_7  beta_kl: 0.2, beta_rec: 0.4, beta_neg: 256, batch_size: 8 ,1/alpha:0.5,  chanel: 32
#brain_8  beta_kl: 0.2, beta_rec: 0.2, beta_neg: 25,  batch_size: 8 ,1/alpha:0.25, chanel: 32
#brain_9  beta_kl: 0.2, beta_rec: 0.4, beta_neg: 25,  batch_size: 8 ,1/alpha:0.5,  chanel: 32
#brain_10 beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 32 /mamagood
#brain_11 beta_kl: 1.0, beta_rec: 1.0, beta_neg: 1024, batch_size: 8 ,1/alpha:0.25, chanel: 32
#brain_12 beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 32 /80*80*80 /verygood
#brain_13 beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, batch_size: 8 ,1/alpha:0.25, chanel: 64 

    beta_kl = 1.0
    beta_rec = 1.0
    beta_neg = 1024
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(device)
    print("betas: ", beta_kl, beta_neg, beta_rec)
    try:
        train_soft_intro_vae(dataset="brain", batch_size=8, num_workers=4, num_epochs=300,
                             num_vae=0, beta_kl=beta_kl, beta_neg=beta_neg, beta_rec=beta_rec,
                             device=device, save_interval=50, start_epoch=0, lr_e=2e-4, lr_d=2e-4,
                             pretrained=None,
                             test_iter=1920, with_fid=False)
    except SystemError:
        print("Error, probably loss is NaN, try again...")
    
    
