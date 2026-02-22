# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import logging

# Suppress BERT/HuggingFace HTTP request spam, warnings, and progress bars
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '0'
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('tqdm').setLevel(logging.WARNING)

import numpy as np
import random
import copy
from torch.backends import cudnn
import Config
from Load_Dataset import (RandomGenerator, ValGenerator, ImageToImage2D, LV2D,
                         RandomGeneratorSemi, create_semi_supervised_datasets)
from nets.LViT import LViT
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, train_one_epoch_semi, print_summary
import Config as config
from torchvision import transforms
from utils import (CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch,
                   SemiSupervisedLoss, get_current_consistency_weight,
                   compute_text_similarity, find_most_similar_text, compute_lv_loss)
from thop import profile

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    train_tf_semi = transforms.Compose([RandomGeneratorSemi(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        
        if config.semi_supervised:
            # Semi-supervised: Split into labeled (25%) and unlabeled (75%)
            labeled_dataset, unlabeled_dataset = create_semi_supervised_datasets(
                config.train_dataset, config.task_name, train_text, train_tf_semi,
                labeled_ratio=config.labeled_ratio, image_size=config.img_size, seed=config.seed
            )
            logger.info(f'Semi-supervised mode: {len(labeled_dataset)} labeled, {len(unlabeled_dataset)} unlabeled')
        else:
            train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                           image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
        
    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        if config.semi_supervised:
            labeled_dataset, unlabeled_dataset = create_semi_supervised_datasets(
                config.train_dataset, config.task_name, text, train_tf_semi,
                labeled_ratio=config.labeled_ratio, image_size=config.img_size, seed=config.seed
            )
            logger.info(f'Semi-supervised mode: {len(labeled_dataset)} labeled, {len(unlabeled_dataset)} unlabeled')
        else:
            train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                           image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size)

    elif config.task_name == 'BTXRD':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        
        if config.semi_supervised:
            labeled_dataset, unlabeled_dataset = create_semi_supervised_datasets(
                config.train_dataset, config.task_name, train_text, train_tf_semi,
                labeled_ratio=config.labeled_ratio, image_size=config.img_size, seed=config.seed
            )
            logger.info(f'Semi-supervised mode: {len(labeled_dataset)} labeled, {len(unlabeled_dataset)} unlabeled')
        else:
            train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                           image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    # Create data loaders
    if config.semi_supervised:
        labeled_loader = DataLoader(labeled_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    worker_init_fn=worker_init_fn,
                                    num_workers=8,
                                    pin_memory=True,
                                    drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      worker_init_fn=worker_init_fn,
                                      num_workers=8,
                                      pin_memory=True,
                                      drop_last=True)
        # Always create train_loader for supervised epochs
        train_loader = DataLoader(labeled_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 worker_init_fn=worker_init_fn,
                                 num_workers=8,
                                 pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 worker_init_fn=worker_init_fn,
                                 num_workers=8,
                                 pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        pretrained_UNet_model_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')
    input = torch.randn(2, 3, 224, 224)
    text = torch.randn(2, 50, 768)
    flops, params = profile(model, inputs=(input, text, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    
    # Choose loss function based on semi-supervised mode
    if config.semi_supervised:
        criterion = SemiSupervisedLoss(dice_weight=0.5, bce_weight=0.5)
        logger.info(f'EPI decay (beta): {config.beta}')
    else:
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    
    
    # Store text embeddings from labeled data for LV Loss (semi-supervised only)
    text_bank = []
    label_bank = []
    epi_pseudo_labels = {}
    if config.semi_supervised:
        for batch, _ in labeled_loader:
            text_bank.append(batch['text'].cpu())
            label_bank.append(batch['label'].cpu())

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))

        # Warmup phase: always supervised (even in semi mode)
        if epoch < config.warmup_epochs:
            logger.info(f'Supervised training (warmup epoch < {config.warmup_epochs})')
            train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        else:
            # Sync EMA at the transition point
            # No EMA sync needed for EPI

            # After warmup: semi-supervised if enabled
            if config.semi_supervised:
                consistency_weight = get_current_consistency_weight(epoch, config.warmup_epochs, config.rampup_length, 0.3)
                logger.info(f'Semi-supervised training (epoch >= {config.warmup_epochs}), Consistency weight: {consistency_weight:.4f}')
                train_one_epoch_semi(
                    labeled_loader, unlabeled_loader, model, criterion, optimizer,
                    writer, epoch, consistency_weight,
                    text_bank, label_bank, model_type, logger,
                    epi_pseudo_labels
                )
            else:
                logger.info(f'Supervised training (epoch >= {config.warmup_epochs})')
                train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
