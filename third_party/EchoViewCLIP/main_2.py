import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from trainers import vificlip_no as vificlip
from utils.loss import TextSemanticOppositeLoss
import matplotlib.pyplot as plt


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    class_names = [class_name for i, class_name in train_data.classes]

    # Custom trainer for different variants of ViFi-CLIP
    model = vificlip.returnCLIP(config,
                                logger=logger,
                                class_names=class_names,)

    model = model.cuda()  # changing to cuda here

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                       smoothing=config.AUG.LABEL_SMOOTH,
                                       mixup_alpha=config.AUG.MIXUP,
                                       cutmix_alpha=config.AUG.CUTMIX,
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = build_optimizer(config, model) # AdamW
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=False)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        if start_epoch > 1:
            logger.info("resetting epochs no and max. accuracy to 0 after loading pre-trained weights")
            start_epoch = 0
            max_accuracy = 0
    if config.TEST.ONLY_TEST:
        acc1 = validate_(val_loader, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            acc1 = validate(val_loader, model, config)
            logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
            is_best = acc1 > max_accuracy
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            if dist.get_rank() == 0 and (
                    epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1) or is_best):
                epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT,
                             is_best)
    # Now doing the multi-view inference crop for videos
    # 4 CLIPs are obtained from each video, and for each CLIP, we get 3 crops (augmentations)
    multi_view_inference = config.TEST.MULTI_VIEW_INFERENCE
    if multi_view_inference:
        config.defrost()
        config.TEST.NUM_CLIP = 4
        config.TEST.NUM_CROP = 3
        config.freeze()
        train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
        acc1 = validate(val_loader, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn):
    model.train()
    optimizer.zero_grad()
    criterion2 = TextSemanticOppositeLoss(mode="L2")

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()


    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        output, output_no, text_features, text_features_no, _, _, _, _ = model(images)

        total_loss = criterion(-output_no, label_id) + criterion2(text_features, text_features_no)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(val_loader, model, config):
    model.eval()

    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter_no, acc5_meter_no = AverageMeter(), AverageMeter()
    with torch.no_grad():
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_no = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                output, output_no, _, _, _, _, _, _ = model(image_input)

                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity

                similarity_no = output_no.view(b, -1).softmax(dim=-1)
                tot_similarity_no += similarity_no
                tot_similarity_no = 1 / tot_similarity_no

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)

            values_1_no, indices_1_no = tot_similarity_no.topk(1, dim=-1)
            values_5_no, indices_5_no = tot_similarity_no.topk(5, dim=-1)

            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)

            acc1_no, acc5_no = 0, 0
            for i in range(b):
                if indices_1_no[i] == label_id[i]:
                    acc1_no += 1
                if label_id[i] in indices_5_no[i]:
                    acc5_no += 1

            acc1_meter_no.update(float(acc1_no) / b * 100, b)
            acc5_meter_no.update(float(acc5_no) / b * 100, b)

            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc@1_no: {acc1_meter_no.avg:.3f}\t'
                    f'Acc@5: {acc5_meter.avg:.3f}\t'
                    f'Acc@5_no: {acc5_meter_no.avg:.3f}\t'
                )

    acc1_meter.sync()
    acc5_meter.sync()
    acc1_meter_no.sync()
    acc5_meter_no.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * Acc@1_no {acc1_meter_no.avg:.3f} Acc@5_no {acc5_meter_no.avg:.3f}')
    return acc1_meter_no.avg

@torch.no_grad()
def validate_(val_loader, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter_no, acc5_meter_no = AverageMeter(), AverageMeter()
    all_preds = []
    all_labels = []
    all_preds_no = []
    all_labels_no = []
    class_correct = torch.zeros(config.DATA.NUM_CLASSES).cuda()
    class_total = torch.zeros(config.DATA.NUM_CLASSES).cuda()
    class_correct_no = torch.zeros(config.DATA.NUM_CLASSES).cuda()
    class_total_no = torch.zeros(config.DATA.NUM_CLASSES).cuda()
    
    logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")

    all_image_features_yes = []
    all_image_features_no = []
    all_text_features_yes = []
    all_text_features_no = []
    all_yes_weights = []
    all_no_weights = []
    for idx, batch_data in enumerate(val_loader):
        _image = batch_data["imgs"]
        label_id = batch_data["label"].reshape(-1)
        
        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)
        
        tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
        tot_similarity_no_s = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
        label_id = label_id.cuda(non_blocking=True)
        
        for i in range(n):
            image = _image[:, i, :, :, :, :].cuda(non_blocking=True)
            
            if config.TRAIN.OPT_LEVEL == 'O2':
                image = image.half()
            
            output, output_no, text_features, text_features_no, image_features, image_features_no, yes_weights, no_weights = model(image)
            similarity = output.view(b, -1).softmax(dim=-1)
            tot_similarity += similarity

            similarity_no = output_no.view(b, -1).softmax(dim=-1)
            tot_similarity_no_s += similarity_no
            tot_similarity_no = 1 / tot_similarity_no_s

            text_features = text_features.view(b, -1)
            text_features_no = text_features_no.view(b, -1)
            image_features = image_features.view(b, -1)
            image_features_no = image_features_no.view(b, -1)
            yes_weights = yes_weights.view(b, -1)
            no_weights = no_weights.view(b, -1)
            
            
        all_yes_weights.append(yes_weights.cpu().numpy())
        all_no_weights.append(no_weights.cpu().numpy())
        all_image_features_yes.append(image_features.cpu().numpy())
        all_image_features_no.append(image_features_no.cpu().numpy())
        all_text_features_yes.append(text_features.cpu().numpy())
        all_text_features_no.append(text_features_no.cpu().numpy())
        
        all_preds.append(tot_similarity.cpu().numpy())
        all_labels.append(label_id.cpu().numpy())

        all_preds_no.append(tot_similarity_no_s.cpu().numpy())
        all_labels_no.append(label_id.cpu().numpy())
        
        values_1, indices_1 = tot_similarity.topk(1, dim=-1)
        values_5, indices_5 = tot_similarity.topk(5, dim=-1)

        values_1_no, indices_1_no = tot_similarity_no.topk(1, dim=-1)
        values_5_no, indices_5_no = tot_similarity_no.topk(5, dim=-1)
        
        acc1, acc5 = 0, 0
        acc1_no, acc5_no = 0, 0
        for i in range(b):
            class_total[label_id[i]] += 1
            if indices_1[i] == label_id[i]:
                acc1 += 1
                class_correct[label_id[i]] += 1
            if label_id[i] in indices_5[i]:
                acc5 += 1

            class_total_no[label_id[i]] += 1
            if indices_1_no[i] == label_id[i]:
                acc1_no += 1
                class_correct_no[label_id[i]] += 1
            if label_id[i] in indices_5_no[i]:
                acc5_no += 1
            
        acc1_meter.update(float(acc1) / b * 100, b)
        acc5_meter.update(float(acc5) / b * 100, b)

        acc1_meter_no.update(float(acc1_no) / b * 100, b)
        acc5_meter_no.update(float(acc5_no) / b * 100, b)
        
        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(val_loader)}]\t'
                f'Acc@1: {acc1_meter.avg:.3f}\t'
                f'Acc@1_no: {acc1_meter_no.avg:.3f}\t'
                f'Acc@5: {acc5_meter.avg:.3f}\t'
                f'Acc@5_no: {acc5_meter_no.avg:.3f}\t'
            )
    
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * Acc@1_no {acc1_meter_no.avg:.3f} Acc@5_no {acc5_meter_no.avg:.3f}')
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds_no = np.concatenate(all_preds_no, axis=0)
    all_labels_no = np.concatenate(all_labels_no, axis=0)

    all_yes_weights = np.concatenate(all_yes_weights, axis=0)
    all_no_weights = np.concatenate(all_no_weights, axis=0)

    all_image_features_yes = np.concatenate(all_image_features_yes, axis=0)
    all_image_features_no = np.concatenate(all_image_features_no, axis=0)
    all_text_features_yes = np.concatenate(all_text_features_yes, axis=0)
    all_text_features_no = np.concatenate(all_text_features_no, axis=0)
    
    precision = precision_score(all_labels, np.argmax(all_preds, axis=1), average='macro')
    recall = recall_score(all_labels, np.argmax(all_preds, axis=1), average='macro')
    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='macro')

    precision_no = precision_score(all_labels_no, np.argmax(all_preds_no, axis=1), average='macro')
    recall_no = recall_score(all_labels_no, np.argmax(all_preds_no, axis=1), average='macro')
    f1_no = f1_score(all_labels_no, np.argmax(all_preds_no, axis=1), average='macro')
    
    logger.info(f' * Precision {precision:.3f} Recall {recall:.3f} F1-score {f1:.3f}')
    logger.info(f' * Precision_no {precision_no:.3f} Recall_no {recall_no:.3f} F1-score_no {f1_no:.3f}')
    
    # Per-class metrics
    for i in range(config.DATA.NUM_CLASSES):
        class_acc = (class_correct[i] / class_total[i]).item() * 100 if class_total[i] > 0 else 0
        precision_i = precision_score(all_labels, np.argmax(all_preds, axis=1), labels=[i], average='macro', zero_division=0)
        recall_i = recall_score(all_labels, np.argmax(all_preds, axis=1), labels=[i], average='macro', zero_division=0)
        f1_i = f1_score(all_labels, np.argmax(all_preds, axis=1), labels=[i], average='macro', zero_division=0)
        logger.info(f'Class {i}: Acc@1 {class_acc:.2f} Precision {precision_i:.3f} Recall {recall_i:.3f} F1-score {f1_i:.3f}')

    # save all labels and correspnding predictions as csv
    np.savetxt(os.path.join(config.OUTPUT, 'all_labels.csv'), all_labels, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(config.OUTPUT, 'all_preds.csv'), all_preds, delimiter=',', fmt='%.6f')

    np.savetxt(os.path.join(config.OUTPUT, 'all_yes_weights.csv'), all_yes_weights, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(config.OUTPUT, 'all_no_weights.csv'), all_no_weights, delimiter=',', fmt='%.6f')

    np.savetxt(os.path.join(config.OUTPUT, 'all_image_features_yes.csv'), all_image_features_yes, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(config.OUTPUT, 'all_image_features_no.csv'), all_image_features_no, delimiter=',', fmt='%.6f')

    np.savetxt(os.path.join(config.OUTPUT, 'all_text_features_yes.csv'), all_text_features_yes, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(config.OUTPUT, 'all_text_features_no.csv'), all_text_features_no, delimiter=',', fmt='%.6f')

    for i in range(config.DATA.NUM_CLASSES):
        class_acc_no = (class_correct_no[i] / class_total_no[i]).item() * 100 if class_total_no[i] > 0 else 0
        precision_i_no = precision_score(all_labels_no, np.argmax(all_preds_no, axis=1), labels=[i], average='macro', zero_division=0)
        recall_i_no = recall_score(all_labels_no, np.argmax(all_preds_no, axis=1), labels=[i], average='macro', zero_division=0)
        f1_i_no = f1_score(all_labels_no, np.argmax(all_preds_no, axis=1), labels=[i], average='macro', zero_division=0)
        logger.info(f'Class {i}: Acc@1_no {class_acc_no:.2f} Precision_no {precision_i_no:.3f} Recall_no {recall_i_no:.3f} F1-score_no {f1_i_no:.3f}')

    np.savetxt(os.path.join(config.OUTPUT, 'all_labels_no.csv'), all_labels_no, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(config.OUTPUT, 'all_preds_no.csv'), all_preds_no, delimiter=',', fmt='%.6f')

    plot_roc_curve(all_labels, all_preds, config.DATA.NUM_CLASSES, config.OUTPUT)
    plot_roc_curve(all_labels_no, all_preds_no, config.DATA.NUM_CLASSES, config.OUTPUT, type='no')
    
    return acc1_meter.avg


def plot_roc_curve(y_true, y_score, num_classes, outdir, type='yes'):
    from sklearn.preprocessing import label_binarize

    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=[i for i in range(num_classes)])

    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Classification')
    plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))

    if type == 'yes':
        outpath = os.path.join(outdir, 'roc_curve_yes.png')
    else:
        outpath = os.path.join(outdir, 'roc_curve_no.png')
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
