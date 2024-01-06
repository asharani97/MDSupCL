"""Adapted from https://github.com/facebookresearch/GDT"""

import os
import sys
import time
import torch
import datetime
from logging import getLogger
from data.pair import final
from accelerate import Accelerator
from model.model_train import GDT
from torch.utils.tensorboard import SummaryWriter
from src.log_utils import MetricLoggerGDT, SmoothedValue
from data.data_uniform_Sup import VideoDataset
from data.sampler import BalancedBatchSampler
from src.scheduler import GradualWarmupScheduler
from src.gdt_helper import compute_feats, collate_feats, get_pos_neg, get_losses
from src.utils import (
    init_distributed_mode,
    init_signal_handler,
    makedir,
    save_checkpoint,
    trigger_job_requeue)


logger = getLogger()
def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12395'

def main(args):
    # Set up mixed precision training
    setup()

    # Make output dir
    if args.output_dir:
        makedir(args.output_dir)

    # Init distributed mode
    if torch.cuda.is_available():
        init_distributed_mode(args)

    # init signal handler
    init_signal_handler()


    # Set up tensorboard
    tbx_path = os.path.join(args.output_dir, 'tensorboard')
    global_rank = args.rank if args.distributed else 0
    is_master = True if global_rank == 0 else False
    if is_master:
        writer = SummaryWriter(tbx_path)
        writer.add_text(
            'args',
            " \n".join(['%s : %s' % (arg, getattr(args, arg)) for arg in vars(args)]), 
            0
        )
    else:
        writer = None

    # Log version information
    logger.info(args)
    logger.info(f"torch version: {torch.__version__}")

    accelerator = Accelerator()
    # Set distributed mode
    device = accelerator.device

    # Set CudNN benchmark
    torch.backends.cudnn.benchmark = True

    # Create model
    logger.info("Creating model")
    model = GDT(
            vid_base_arch=args.vid_base_arch, 
            vid2_base_arch=args.vid2_base_arch,
            pretrained=False, 
            norm_feat=args.norm_feat, 
            use_mlp=args.use_mlp,
            num_classes=256,
        )
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    model_without_ddp = model
    if args.distributed:
        ngpus_per_node = torch.device_count()
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    # Set up training optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )


    # Set up LR scheduler
    milestones = [int(lr) - args.lr_warmup_epochs for lr in args.lr_milestones.split(',')]
    lr_scheduler = None
    if args.use_scheduler:
        if args.lr_warmup_epochs > 0:
            if args.scheduler_type == 'multi_step':
                logger.info(f'Using Multi-Step LR scheduler')
                scheduler_step = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma
                )
            else:
                logger.info(f'Using Cosine Annealing LR scheduler')
                scheduler_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            lr_scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=args.world_size,
                total_epoch=args.lr_warmup_epochs,
                after_scheduler=scheduler_step
            )
        else:
            if args.scheduler_type == 'multi_step':
                logger.info(f'Using Multi-Step LR scheduler')
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma
                )
            else:
                logger.info(f'Using Cosine Annealing LR scheduler')
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Checkpointing restart
    ckp_path = os.path.join(args.output_dir, 'checkpoints_1', 'checkpoint.pth')
    if os.path.isfile(ckp_path):
        logger.info(f'Loading checkpoint')
        checkpoint = torch.load(ckp_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        logger.info(f'Restrating at epoch {args.start_epoch}')


    ds1 = VideoDataset(dataset='modal1',split='train')
    ds2 = VideoDataset(dataset='modal2',split='train')
    balanced_batch_sampler = BalancedBatchSampler(ds1, 2, 3)
    balanced_batch_sampler1 = BalancedBatchSampler(ds2, 2, 3) 
    print("Creating data loaders", flush=True)
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)

    data_loader_d1 = torch.utils.data.DataLoader(ds1, batch_sampler = balanced_batch_sampler)
    data_loader_d2 = torch.utils.data.DataLoader(ds2, batch_sampler = balanced_batch_sampler1)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if writer:
            writer.add_scalar('train/epoch', epoch, epoch)
        logger.info(f'Start training epoch: {epoch}')
        loss = train_one_epoch(
            args,
            accelerator,
            data_loader_d1,
            data_loader_d2,
            model,
            optimizer,
            device,
            epoch,
            args.print_freq,
            lr_scheduler,
            args.apex,
            writer=writer,
        )
        if lr_scheduler:
            lr_scheduler.step()
        if args.output_dir:
            save_checkpoint(args, epoch, model, optimizer, lr_scheduler)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')


def train_one_epoch(
        args,
        accelerator,
        data_loader_d1,
        data_loader_d2,
        model,
        optimizer,
        device,
        epoch,
        print_freq,
        lr_scheduler,
        apex=False,
        logger=None,
        writer=None,
):

    model.train()
    metric_logger = MetricLoggerGDT(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    dataloader_iterator = iter(data_loader_d1)
    for batch_idx, batch in metric_logger.log_every(data_loader_d2, print_freq, header, logger, writer, 'train', epoch=epoch):
        video1,label1= batch
        if torch.cuda.is_available():
            try:
                (video2, labels2) = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(data_loader_d1)
                (video2, labels2) = next(dataloader_iterator)

        video1, video2 = final(video1,video2,label1,labels2)
        video, v =video1,video2
        start_time = time.time()
        video, v = video.to(device) , v.to(device)

        # form positive and negative pairs dependent on hypothesis
        hyp = 'basecase'

        # compute features
        feats1 = compute_feats(model, video, v)
        feat_v, feat_v2 = feats1

        # collation across GPUs
        feat_v_col, feat_v2_col = collate_feats([feat_v, feat_v2]) #in list
        feats1_col = (feat_v_col, feat_v2_col)

        # basecase cross-modal loss #########################################
        pairs1 = get_pos_neg(hyp, feats1, feats1_col)
        # (V1, V2)
        loss1, loss_dict1 = get_losses(pairs1, nce_t=args.nce_t)
        loss = loss1
        #####################################################################
        loss_dict2 = None

        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        batch_size = video.shape[0]

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['batch_t/s'].update((time.time() - start_time))
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
    torch.cuda.empty_cache()
    return metric_logger.loss.avg


def parse_args():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
                         'Instead, it is %s.' % v)

    import argparse
    parser = argparse.ArgumentParser(description='Video Representation Learning(Multimodal-Setup)')
    parser.register('type', 'bool', str2bool)

    # Data
    parser.add_argument('--workers', default=6, type=int,
        help='number of data loading workers (default: 0)')

    # GDT NCE loss
    parser.add_argument('--nce_t', default=0.07, type=float, 
        help='softmax weighting')

    # Model
    parser.add_argument('--vid_base_arch', default='r2plus1d_18',
        help='Video Base Arch for A-V model',
        choices=['r2plus1d_18', 'r2plus1d_34'])
    parser.add_argument('--vid2_base_arch', default='r2plus1d_18',
        help='Vide Base Arch for A-V model',
        choices=['r2plus1d_18', 'r2plus1d_34'])
    parser.add_argument('--pretrained', default='False', type='bool',
        help='If the modei used is pretrained or not')
    parser.add_argument('--use_mlp', default='True', type='bool',
        help='Use MLP projection head')

    # Training
    
    parser.add_argument('--epochs', default=200, type=int,
        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
        help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='weight decay (default: 1e-5)')
    parser.add_argument('--use_scheduler', default='True', type='bool',
        help='Use LR scheduler')
    parser.add_argument('--scheduler_type', default='multi_step', type=str,
        choices=['multi_step', 'cosine'],
        help='Type of LR scheduler')
    parser.add_argument('--lr_milestones', default='150,175', type=str,
        help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_warmup_epochs', default=10, type=int,
        help='number of warmup epochs')
    parser.add_argument('--norm_feat', default='True', type='bool',
        help='Normalize embeddings')

    # Logging
    parser.add_argument('--print_freq', default=10, type=int,
        help='print frequency')
    parser.add_argument('--output_dir', default='.',
        help='path where to save')

    # Checkpointing
    parser.add_argument('--start_epoch', default=1, type=int,
        help='start epoch')

    # Mixed precision training parameters
    parser.add_argument('--apex', default='False', type='bool', 
        help='Use apex for mixed precision training'
    )

    # distributed training parameters
    parser.add_argument('--distributed', default='False', type='bool',
        help='ddp mode')
    parser.add_argument('--world_size', default=1, type=int,
        help='number of distributed processes')
    parser.add_argument('--master_port', default=-1, type=int,
        help='Master port of Job')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # set multi-processing start method
    import torch.multiprocessing as mp

    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass

    main(args)
