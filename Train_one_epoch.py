# -*- coding: utf-8 -*-
import torch.optim
import torch
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        if text.shape[1] > 50:
            text = text[:, :50, :]
        
        images, masks, text = images.cuda(), masks.cuda(), text.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds = model(images, text)
        out_loss = criterion(preds, masks.float())  # Loss

        # Debug: print statistics of preds and masks
        if i == 1 and epoch % 5 == 0:  # Print once per 5 epochs, first batch only
            preds_np = preds.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            logger.info(f"[DEBUG] preds: min={preds_np.min():.4f}, max={preds_np.max():.4f}, mean={preds_np.mean():.4f}")
            logger.info(f"[DEBUG] masks: min={masks_np.min():.4f}, max={masks_np.max():.4f}, mean={masks_np.mean():.4f}")
        # print(model.training)


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg


##################################################################################
#=================================================================================
#          Semi-Supervised Train One Epoch (EPI + LV Loss)
#=================================================================================
##################################################################################
def train_one_epoch_semi(labeled_loader, unlabeled_loader, model, criterion, optimizer, 
                         writer, epoch, consistency_weight,
                         text_bank, label_bank, model_type, logger,
                         epi_pseudo_labels):
    logging_mode = 'Train'
    end = time.time()
    time_sum, loss_sum = 0, 0
    loss_sup_sum, loss_unsup_sum, loss_lv_sum = 0, 0, 0
    dice_sum, iou_sum = 0.0, 0.0
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    num_iterations = max(len(labeled_loader), len(unlabeled_loader))
    

    beta = config.beta if hasattr(config, 'beta') else 0.99

    for i in range(1, num_iterations + 1):
        try:
            labeled_batch, labeled_names = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            labeled_batch, labeled_names = next(labeled_iter)

        images_l = labeled_batch['image'].cuda()
        masks_l = labeled_batch['label'].cuda()
        text_l = labeled_batch['text'].cuda()
        # Ensure text_l shape is consistent with main train_one_epoch (truncate to 50 if needed)
        if text_l.shape[1] > 50:
            text_l = text_l[:, :50, :]

        try:
            unlabeled_batch, unlabeled_names = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_batch, unlabeled_names = next(unlabeled_iter)

        images_u = unlabeled_batch['image'].cuda()
        text_u = unlabeled_batch['text'].cuda()
        # Ensure text_u shape is consistent with main train_one_epoch (truncate to 50 if needed)
        if text_u.shape[1] > 50:
            text_u = text_u[:, :50, :]

        if i <= len(labeled_loader):
            text_bank.append(text_l.detach().cpu())
            label_bank.append(masks_l.detach().cpu())
            if len(text_bank) > 50:
                text_bank.pop(0)
                label_bank.pop(0)

        preds_l = model(images_l, text_l)
        preds_u = model(images_u, text_u)

        # EPI: Update pseudo-labels for each unlabeled image in the batch
        pseudo_labels = torch.zeros_like(preds_u)
        for idx, name in enumerate(unlabeled_names):
            curr_pred = preds_u[idx:idx+1].detach()
            if name in epi_pseudo_labels:
                prev_pseudo = epi_pseudo_labels[name]
                new_pseudo = beta * prev_pseudo + (1 - beta) * curr_pred
                epi_pseudo_labels[name] = new_pseudo.detach()
            else:
                epi_pseudo_labels[name] = curr_pred.detach()
            pseudo_labels[idx:idx+1] = epi_pseudo_labels[name]
        
        lv_loss = torch.tensor(0.0, device=images_l.device)
        if len(text_bank) > 0:
            curr_text_bank = torch.cat(text_bank, dim=0).cuda() 
            curr_label_bank = torch.cat(label_bank, dim=0).cuda()
            
            indices, similarities = find_most_similar_text(text_u, curr_text_bank, top_k=1)
            
            if indices is not None:
                batch_indices = indices[:, 0]
                contrastive_labels = curr_label_bank[batch_indices].unsqueeze(1).float()
                
                if contrastive_labels.shape[-2:] != preds_u.shape[-2:]:
                    contrastive_labels = F.interpolate(
                        contrastive_labels, size=preds_u.shape[-2:], mode='nearest'
                    )
                
                lv_loss = compute_lv_loss(preds_u, contrastive_labels, similarities)
                del curr_text_bank, curr_label_bank, contrastive_labels

        total_loss, loss_sup, loss_unsup, loss_lv = criterion(
            preds_l, masks_l.float(),
            pred_unlabeled=preds_u,
            pseudo_labels=pseudo_labels,
            lv_loss=lv_loss,
            consistency_weight=consistency_weight
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # No EMA update: EPI handles pseudo-label averaging
        
        # ====================================================
        #             Metrics and Logging
        # ====================================================
        train_dice = criterion._show_dice(preds_l, masks_l.float())
        train_iou = iou_on_batch(masks_l, preds_l)
        
        batch_time = time.time() - end
        
        time_sum += len(images_l) * batch_time
        loss_sum += len(images_l) * total_loss.item()
        loss_sup_sum += len(images_l) * loss_sup.item()
        loss_unsup_sum += len(images_l) * loss_unsup.item()
        loss_lv_sum += len(images_l) * lv_loss.item()
        iou_sum += len(images_l) * train_iou
        dice_sum += len(images_l) * train_dice
        
        if i == num_iterations:
            n_samples = config.batch_size * (i - 1) + len(images_l)
            average_loss = loss_sum / n_samples
            average_time = time_sum / n_samples
            train_iou_average = iou_sum / n_samples
            train_dice_avg = dice_sum / n_samples
            avg_loss_sup = loss_sup_sum / n_samples
            avg_loss_unsup = loss_unsup_sum / n_samples
            avg_loss_lv = loss_lv_sum / n_samples
        else:
            n_samples = i * config.batch_size
            average_loss = loss_sum / n_samples
            average_time = time_sum / n_samples
            train_iou_average = iou_sum / n_samples
            train_dice_avg = dice_sum / n_samples
            avg_loss_sup = loss_sup_sum / n_samples
            avg_loss_unsup = loss_unsup_sum / n_samples
            avg_loss_lv = loss_lv_sum / n_samples
        
        end = time.time()
        torch.cuda.empty_cache()
        
        if i % config.print_frequency == 0:
            # Extended logging for semi-supervised
            summary_str = f'   [{logging_mode}] Epoch: [{epoch + 1}][{i}/{num_iterations}]  '
            summary_str += f'Loss:{total_loss.item():.3f} (Avg {average_loss:.4f}) '
            summary_str += f'L_sup:{loss_sup.item():.3f} L_unsup:{loss_unsup.item():.3f} L_LV:{lv_loss.item():.3f} '
            summary_str += f'IoU:{train_iou:.3f} (Avg {train_iou_average:.4f}) '
            summary_str += f'Dice:{train_dice:.4f} (Avg {train_dice_avg:.4f}) '
            summary_str += f'LR {min(g["lr"] for g in optimizer.param_groups):.2e} '
            summary_str += f'(AvgTime {average_time:.1f})'
            logger.info(summary_str)
        
        if config.tensorboard:
            step = epoch * num_iterations + i
            writer.add_scalar(logging_mode + '_total_loss', total_loss.item(), step)
            writer.add_scalar(logging_mode + '_loss_sup', loss_sup.item(), step)
            writer.add_scalar(logging_mode + '_loss_unsup', loss_unsup.item(), step)
            writer.add_scalar(logging_mode + '_loss_lv', lv_loss.item(), step)
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)
            writer.add_scalar(logging_mode + '_consistency_weight', consistency_weight, step)
        
        torch.cuda.empty_cache()
    
    logger.info(f'   Epoch Summary: L_sup={avg_loss_sup:.4f}, L_unsup={avg_loss_unsup:.4f}, L_LV={avg_loss_lv:.4f}')
    
    return average_loss, train_dice_avg
    