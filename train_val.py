# --------------------------------------------------------
# CGNL Network
# Copyright (c) 2018 Kaiyu Yue
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import argparse
import time
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel

from torchvision import transforms
from termcolor import cprint
from lib import dataloader
from model import resnet

# torch version
cprint('=> Torch Vresion: ' + torch.__version__, 'green')

# args
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--debug', '-d', dest='debug', action='store_true',
        help='enable debug mode')
parser.add_argument('--warmup', '-w', dest='warmup', action='store_true',
        help='using warmup strategy')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
        help='print frequency (default: 10)')
parser.add_argument('--nl-nums', default=0, type=int, metavar='N',
        help='number of the NL | CGNL block (default: 0)')
parser.add_argument('--nl-type', default=None, type=str,
        help='choose NL | CGNL | CGNLx block to add (default: None)')
parser.add_argument('--arch', default='50', type=str,
        help='the depth of resnet (default: 50)')
parser.add_argument('--valid', '-v', dest='valid',
        action='store_true', help='just run validation')
parser.add_argument('--checkpoints', default='', type=str,
        help='the dir of checkpoints')
parser.add_argument('--dataset', default='cub', type=str,
        help='cub | imagenet (default: cub)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
        help='initial learning rate (default: 0.01)')

best_prec1 = 0
best_prec5 = 0

def main():
    global args
    global best_prec1, best_prec5

    args = parser.parse_args()

    # simple args
    debug = args.debug
    if debug: cprint('=> WARN: Debug Mode', 'yellow')

    dataset = args.dataset
    num_classes = 200 if dataset == 'cub' else 1000
    base_size = 512 if dataset == 'cub' else 256
    pool_size = 14 if base_size == 512 else 7
    workers = 0 if debug else 8
    batch_size = 2 if debug else 256
    if base_size == 512 and \
        args.arch == '152':
        batch_size = 128
    drop_ratio = 0.1
    lr_drop_epoch_list = [31, 61, 81]
    epochs = 100
    eval_freq = 1
    gpu_ids = [0] if debug else [0,1,2,3,4,5,6,7]
    crop_size = 224 if base_size == 256 else 448

    # args for the nl and cgnl block
    arch = args.arch
    nl_type  = args.nl_type # 'cgnl' | 'cgnlx' | 'nl'
    nl_nums  = args.nl_nums # 1: stage res4

    # warmup setting
    WARMUP_LRS = [args.lr * (drop_ratio**len(lr_drop_epoch_list)), args.lr]
    WARMUP_EPOCHS = 10

    # data loader
    if dataset == 'cub':
        data_root = 'data/cub'
        imgs_fold = os.path.join(data_root, 'images')
        train_ann_file = os.path.join(data_root, 'cub_train.list')
        valid_ann_file = os.path.join(data_root, 'cub_val.list')
    elif dataset == 'imagenet':
        data_root = 'data/imagenet'
        imgs_fold = os.path.join(data_root)
        train_ann_file = os.path.join(data_root, 'imagenet_train.list')
        valid_ann_file = os.path.join(data_root, 'imagenet_val.list')
    else:
        raise NameError("WARN: The dataset '{}' is not supported yet.")

    train_dataset = dataloader.ImgLoader(
            root = imgs_fold,
            ann_file = train_ann_file,
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=crop_size, scale=(0.08, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
                ]))

    val_dataset = dataloader.ImgLoader(
            root = imgs_fold,
            ann_file = valid_ann_file,
            transform = transforms.Compose([
                transforms.Resize(base_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
                ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = workers,
            pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = workers,
            pin_memory = True)

    # build model
    model = resnet.model_hub(arch,
                             pretrained=True,
                             nl_type=nl_type,
                             nl_nums=nl_nums,
                             pool_size=pool_size)

    # change the fc layer
    model._modules['fc'] = torch.nn.Linear(in_features=2048,
                                           out_features=num_classes)
    torch.nn.init.kaiming_normal_(model._modules['fc'].weight,
                                  mode='fan_out', nonlinearity='relu')
    print(model)

    # parallel
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=1e-4)

    # cudnn
    cudnn.benchmark = True

    # warmup
    if args.warmup:
        epochs += WARMUP_EPOCHS
        lr_drop_epoch_list = list(
                np.array(lr_drop_epoch_list) + WARMUP_EPOCHS)
        cprint('=> WARN: warmup is used in the first {} epochs'.format(
            WARMUP_EPOCHS), 'yellow')

    # valid
    if args.valid:
        cprint('=> WARN: Validation Mode', 'yellow')
        print('start validation ...')
        checkpoint_fold = args.checkpoints
        checkpoint_best = os.path.join(checkpoint_fold, 'model_best.pth.tar')
        print('=> loading state_dict from {}'.format(checkpoint_best))
        model.load_state_dict(
                torch.load(checkpoint_best)['state_dict'])
        prec1, prec5 = validate(val_loader, model, criterion)
        print(' * Final Accuracy: Prec@1 {:.3f}, Prec@5 {:.3f}'.format(prec1, prec5))
        exit(0)

    # train
    print('start training ...')
    for epoch in range(0, epochs):
        current_lr = adjust_learning_rate(optimizer, drop_ratio, epoch, lr_drop_epoch_list,
                                          WARMUP_EPOCHS, WARMUP_LRS)
        # train one epoch
        train(train_loader, model, criterion, optimizer, epoch, epochs, current_lr)

        if nl_nums > 0:
            checkpoint_name = '{}-r-{}-w-{}{}-block.pth.tar'.format(dataset, arch, nl_nums, nl_type)
        else:
            checkpoint_name = '{}-r-{}-base.pth.tar'.format(dataset, arch)

        if (epoch + 1) % eval_freq == 0:
            prec1, prec5 = validate(val_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            print(' * Best accuracy: Prec@1 {:.3f}, Prec@5 {:.3f}'.format(best_prec1, best_prec5))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=checkpoint_name)


def train(train_loader, model, criterion, optimizer, epoch, epochs, current_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0:3d}/{1:3d}][{2:3d}/{3:3d}]\t'
                  'LR: {lr:.7f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, epochs, i, len(train_loader), 
                   lr=current_lr, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def adjust_learning_rate(optimizer, drop_ratio, epoch, lr_drop_epoch_list,
                         WARMUP_EPOCHS, WARMUP_LRS):
    if args.warmup and epoch < WARMUP_EPOCHS:
        # achieve the warmup lr
        lrs = np.linspace(WARMUP_LRS[0], WARMUP_LRS[1], num=WARMUP_EPOCHS)
        cprint('=> warmup lrs {}'.format(lrs), 'green')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[epoch]
        current_lr = lrs[epoch]
    else:
        decay = drop_ratio if epoch in lr_drop_epoch_list else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * decay
        args.lr *= decay
        current_lr = args.lr
    return current_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
