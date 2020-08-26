# -*- coding: UTF-8 -*-

import os
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import layers
import models
import datasets
import utils
import metrics
import visualization
from itertools import chain
import torchvision


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-type', type=str, required=True,
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'STL10', 'ImageNet10'],
                        help='type of the dataset')
    parser.add_argument('--dataset-path', type=str, required=True, help='path to the dataset')

    parser.add_argument('--img-type', type=str, default='grayscale', choices=['rgb', 'grayscale', 'sobel'],
                        help='type of the image')

    parser.add_argument('--dim-zs', type=int, default=50, help='dimension of zs')
    parser.add_argument('--dim-zc', type=int, default=10, help='dimension of zc')
    parser.add_argument('--zs-std', type=float, default=0.1,
                        help='standard deviation of the prior gaussian distribution for zs')

    parser.add_argument('--beta-mi', type=float, default=0.5, help='beta mi')
    parser.add_argument('--beta-adv', type=float, default=1., help='beta adv')
    parser.add_argument('--beta-aug', type=float, default=2., help='beta aug')

    parser.add_argument('--lambda-gp', type=float, default=10.0, help='gradient penalty coefficient')
    parser.add_argument('--skip-iter', type=int, default=4,
                        help='the number of critic iterations per encoder iteration')

    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs, note that you can early stop when the critic loss converges')

    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers for the dataloaders')
    parser.add_argument('--checkpoint-root', type=str, default='./checkpoint', help='path to the checkpoint root')
    parser.add_argument('--save-per-epochs', type=int, default=50, help='save the models per number of epochs')
    parser.add_argument('--model-name', type=str, default='DCCS', help='name of the model')

    args = parser.parse_args()

    # create checkpoint directory
    # checkpoint_root/dataset_type/model_name/
    checkpoint_path = os.path.join(args.checkpoint_root, args.dataset_type, args.model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    # directory to save models
    os.makedirs(os.path.join(checkpoint_path, 'model'), exist_ok=True)
    # directory to save images
    os.makedirs(os.path.join(checkpoint_path, 'img'), exist_ok=True)

    # create logger
    console_logger, file_logger = utils.create_logger(os.path.join(checkpoint_path, 'train.log'))

    file_logger.info('Args: %s' % str(args))
    file_logger.info('Checkpoint path: %s' % checkpoint_path)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create datasets
    train_dataset = datasets.ClusterDataset(args.dataset_path, args.dataset_type, args.img_type, training=True)
    eval_dataset = datasets.ClusterDataset(args.dataset_path, args.dataset_type, args.img_type, training=False)

    file_logger.info('Number of training samples: %d' % len(train_dataset))
    file_logger.info('Number of evaluating samples: %d' % len(eval_dataset))

    file_logger.info('Transforms for the images: %s' % str(train_dataset.transforms))
    file_logger.info('Transforms for the augmented images: %s' % str(train_dataset.transforms_aug))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    # create models
    encoder = models.get_encoder(args.dataset_type, args.img_type, args.dim_zs, args.dim_zc)
    critic = models.get_critic(args.dim_zs, args.dim_zc)
    discriminator = models.get_discriminator(args.dataset_type, args.dim_zs, args.dim_zc)
    sobel = layers.SobelLayer(normalize=True)

    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            encoder = nn.DataParallel(encoder)
            critic = nn.DataParallel(critic)
            discriminator = nn.DataParallel(discriminator)
            sobel = nn.DataParallel(sobel)
        file_logger.info('Using %d GPU' % num_gpus)
    else:
        device = torch.device('cpu')
        file_logger.info('Using CPU')
    encoder.to(device)
    critic.to(device)
    discriminator.to(device)
    sobel.to(device)

    # create optimizers
    optimizer_e = optim.Adam(chain(encoder.parameters(), discriminator.parameters()), lr=args.lr, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # create SummaryWriter
    writer = SummaryWriter(comment='_' + '_'.join([args.dataset_type, args.model_name]))

    max_acc = 0
    max_nmi = 0
    max_ari = 0
    global_step = 0
    for epoch in range(args.epochs):
        # train
        global_step = train_epoch(train_loader, encoder, critic, discriminator, sobel, device,
                                  optimizer_e, optimizer_c, epoch, global_step, file_logger, writer, args)

        # eval
        max_acc, max_nmi, max_ari = eval_epoch(eval_loader, encoder, critic, discriminator, sobel,
                                               device, epoch, checkpoint_path, file_logger, writer,
                                               (max_acc, max_nmi, max_ari), args)

    writer.close()


def train_epoch(train_loader, encoder, critic, discriminator, sobel, device, optimizer_e, optimizer_c, epoch,
                global_step, file_logger, writer, args):
    train_data_time = utils.AverageMeter()
    train_batch_time = utils.AverageMeter()
    train_mi_loss = utils.AverageMeter()
    train_aug_loss = utils.AverageMeter()
    train_adv_e_loss = utils.AverageMeter()
    train_adv_c_loss = utils.AverageMeter()

    bce_loss = nn.BCEWithLogitsLoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    encoder.train()
    critic.train()
    discriminator.train()

    tic = time.time()
    for data in train_loader:
        train_data_time.update(time.time() - tic)

        x, x_aug = data
        x = x.to(device, non_blocking=True)
        x_aug = x_aug.to(device, non_blocking=True)
        if args.img_type == 'sobel':
            x = sobel(x)
            x_aug = sobel(x_aug)

        b = x.size(0)

        # train encoder and discriminator
        if global_step % (args.skip_iter + 1) == args.skip_iter:
            # calculate zc_aug_logit
            if args.beta_aug != 0:
                _, zc_aug_logit, _ = encoder(x_aug)
            # calculate z
            zs, zc_logit, dis_x = encoder(x)
            zc = F.softmax(zc_logit, dim=1)
            z = torch.cat([zs, zc], dim=1)

            # adv e loss
            adv_e_loss = args.beta_adv * -torch.mean(critic(z))

            # mi loss
            if args.beta_mi > 0:
                dis_label = torch.zeros((b * 2, 1), dtype=torch.float32, device=device)
                dis_label[:b].fill_(1)

                z_bar = z[torch.randperm(b)]
                concat_x = torch.cat([dis_x, dis_x], dim=0)
                concat_z = torch.cat([z, z_bar], dim=0)
                dis_logit = discriminator(concat_x, concat_z)
                mi_loss = args.beta_mi * bce_loss(dis_logit, dis_label)
            else:
                mi_loss = torch.tensor(0, dtype=torch.float32, device=device)

            # aug loss
            if args.beta_aug != 0:
                aug_loss = args.beta_aug * kl_div_loss(F.log_softmax(zc_aug_logit, dim=1), zc)
            else:
                aug_loss = torch.tensor(0, dtype=torch.float32, device=device)

            e_loss = adv_e_loss + mi_loss + aug_loss

            optimizer_e.zero_grad()
            e_loss.backward()
            optimizer_e.step()

            train_adv_e_loss.update(adv_e_loss.item(), n=b)
            train_mi_loss.update(mi_loss.item(), n=b * 2)
            train_aug_loss.update(aug_loss.item(), n=b)
        # train critic
        else:
            with torch.no_grad():
                zs, zc_logit, _ = encoder(x)
                zc = F.softmax(zc_logit, dim=1)
                z = torch.cat([zs, zc], dim=1)
            zs_prior, zc_prior, _ = utils.sample_z(b, dim_zs=args.dim_zs, dim_zc=args.dim_zc, zs_std=args.zs_std)
            zs_prior = torch.tensor(zs_prior, dtype=torch.float32, device=device)
            zc_prior = torch.tensor(zc_prior, dtype=torch.float32, device=device)
            z_prior = torch.cat([zs_prior, zc_prior], dim=1)

            c_real_loss = -torch.mean(critic(z_prior))
            c_fake_loss = torch.mean(critic(z))
            gradient_penalty = utils.calc_gradient_penalty(critic, z_prior, z, args.lambda_gp)

            adv_c_loss = args.beta_adv * (c_real_loss + c_fake_loss + gradient_penalty)

            optimizer_c.zero_grad()
            adv_c_loss.backward()
            optimizer_c.step()

            train_adv_c_loss.update(adv_c_loss.item(), n=b)

        train_batch_time.update(time.time() - tic)

        global_step += 1
        tic = time.time()

    file_logger.info('Epoch {0} (train):\t'
                     'data_time: {data_time.sum:.2f}s\t'
                     'batch_time: {batch_time.sum:.2f}s\t'
                     'mi_loss: {mi_loss.avg:.4f}\t'
                     'aug_loss: {aug_loss.avg:.4f}\t'
                     'adv_e_loss: {adv_e_loss.avg:.4f}\t'
                     'adv_c_loss: {adv_c_loss.avg:.4f}\t'.format(
        epoch, data_time=train_data_time, batch_time=train_batch_time, mi_loss=train_mi_loss,
        aug_loss=train_aug_loss, adv_e_loss=train_adv_e_loss, adv_c_loss=train_adv_c_loss))

    writer.add_scalars('mi_loss', {'train': train_mi_loss.avg}, epoch)
    writer.add_scalars('aug_loss', {'train': train_aug_loss.avg}, epoch)
    writer.add_scalars('adv_e_loss', {'train': train_adv_e_loss.avg}, epoch)
    writer.add_scalars('adv_c_loss', {'train': train_adv_c_loss.avg}, epoch)

    return global_step


def eval_epoch(eval_loader, encoder, critic, discriminator, sobel, device, epoch, checkpoint_path, file_logger, writer,
               best_metrics, args):
    max_acc, max_nmi, max_ari = best_metrics

    eval_data_time = utils.AverageMeter()
    eval_batch_time = utils.AverageMeter()

    imgs = list()
    zs = list()
    zc_logit = list()
    zc = list()
    y_true = list()

    encoder.eval()

    tic = time.time()
    with torch.no_grad():
        for data in eval_loader:
            eval_data_time.update(time.time() - tic)
            x, y_true_ = data
            x = x.to(device, non_blocking=True)
            if args.img_type == 'sobel':
                x = sobel(x)

            zs_, zc_logit_, _ = encoder(x)
            zc_ = F.softmax(zc_logit_, dim=1)

            imgs.append(x.cpu().numpy())
            zs.append(zs_.cpu().numpy())
            zc_logit.append(zc_logit_.cpu().numpy())
            zc.append(zc_.cpu().numpy())
            y_true.append(y_true_.cpu().numpy())

            eval_batch_time.update(time.time() - tic)

            tic = time.time()

    imgs = np.concatenate(imgs, axis=0)
    zs = np.concatenate(zs, axis=0)
    zc_logit = np.concatenate(zc_logit, axis=0)
    zc = np.concatenate(zc, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # calculate metrics
    y_pred = np.argmax(zc_logit, axis=1)

    num_classes = zc_logit.shape[1]
    match = utils.hungarian_match(y_pred, y_true, num_classes)
    y_pred = utils.convert_cluster_assignment_to_ground_truth(y_pred, match)

    acc = metrics.accuracy(y_pred, y_true)
    nmi = metrics.nmi(y_pred, y_true)
    ari = metrics.ari(y_pred, y_true)

    max_acc = max(max_acc, acc)
    max_nmi = max(max_nmi, nmi)
    max_ari = max(max_ari, ari)

    tic = time.time()
    # save some images
    if epoch == 0:
        real_img_idx = np.random.choice(np.arange(len(eval_loader.dataset)), 100, replace=False)
        imgs_ = torch.tensor(imgs[real_img_idx], dtype=torch.float32)
        if args.img_type == 'sobel':
            torchvision.utils.save_image(imgs_[:, :1], os.path.join(checkpoint_path, 'img', 'real_x.jpg'), nrow=10,
                                         padding=0, normalize=True, range=(-1, 1))
            torchvision.utils.save_image(imgs_[:, 1:], os.path.join(checkpoint_path, 'img', 'real_y.jpg'), nrow=10,
                                         padding=0, normalize=True, range=(-1, 1))
        else:
            torchvision.utils.save_image(imgs_, os.path.join(checkpoint_path, 'img', 'real.jpg'), nrow=10,
                                         padding=0, normalize=True, range=(-1, 1))
    if epoch == 0 or (epoch + 1) % args.save_per_epochs == 0:
        # save models
        utils.save_model(encoder, os.path.join(checkpoint_path, 'model', 'encoder_%03d.tar' % epoch))
        utils.save_model(critic, os.path.join(checkpoint_path, 'model', 'critic_%03d.tar' % epoch))
        utils.save_model(discriminator, os.path.join(checkpoint_path, 'model', 'discriminator_%03d.tar' % epoch))

        # save top 10 images for each cluster
        cluster_imgs = list()
        for cls in range(num_classes):
            cls_score = zc[:, cls]
            idxs = np.argsort(cls_score)[::-1][:10]
            cluster_imgs.append(imgs[idxs])
        cluster_imgs = np.concatenate(cluster_imgs, axis=0)
        cluster_imgs = torch.tensor(cluster_imgs, dtype=torch.float32)
        if args.img_type == 'sobel':
            torchvision.utils.save_image(cluster_imgs[:, :1],
                                         os.path.join(checkpoint_path, 'img', 'cluster_imgs_%03d_x.jpg' % epoch),
                                         nrow=10, padding=0, normalize=True, range=(-1, 1))
            torchvision.utils.save_image(cluster_imgs[:, 1:],
                                         os.path.join(checkpoint_path, 'img', 'cluster_imgs_%03d_y.jpg' % epoch),
                                         nrow=10, padding=0, normalize=True, range=(-1, 1))
        else:
            torchvision.utils.save_image(cluster_imgs,
                                         os.path.join(checkpoint_path, 'img', 'cluster_imgs_%03d.jpg' % epoch), nrow=10,
                                         padding=0, normalize=True, range=(-1, 1))

        # save tsne image
        idxs = np.random.choice(np.arange(len(eval_loader.dataset)), 1000, replace=False)
        z = np.concatenate([zs[idxs], zc[idxs]], axis=1)
        visualization.tsne(z, y=y_true[idxs], show_legend=False,
                           save_path=os.path.join(checkpoint_path, 'img', 'tsne_%03d.jpg' % epoch), show_fig=False)
    eval_save_time = time.time() - tic

    file_logger.info('Epoch {0} (eval):\t'
                     'data_time: {data_time.sum:.2f}s\t'
                     'batch_time: {batch_time.sum:.2f}s\t'
                     'save_time: {save_time:.2f}s\t'
                     'acc: {acc:.2f}% ({max_acc:.2f}%)\t'
                     'nmi: {nmi:.4f} ({max_nmi:.4f})\t'
                     'ari: {ari:.4f} ({max_ari:.4f})\t'.format(epoch,
                                                               data_time=eval_data_time, batch_time=eval_batch_time,
                                                               save_time=eval_save_time,
                                                               acc=acc, max_acc=max_acc, nmi=nmi, max_nmi=max_nmi,
                                                               ari=ari, max_ari=max_ari))
    writer.add_scalars('acc', {'val': acc}, epoch)
    writer.add_scalars('nmi', {'val': nmi}, epoch)
    writer.add_scalars('ari', {'val': ari}, epoch)

    return max_acc, max_nmi, max_ari


if __name__ == '__main__':
    _main()
