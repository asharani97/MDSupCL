"""Adapted from https://github.com/facebookresearch/GDT"""

import os
import sys
import time
import torch
import datetime
import torchvision
import torch.nn as nn
from statistics import mean
from accelerate import Accelerator
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from model.model import load_model, Identity
from data.data_test import VideoDataset
# Custom imports
from src.scheduler import GradualWarmupScheduler
from utils import (
    AverageMeter,
    accuracy,
    aggregrate_video_accuracy,
    initialize_exp,
    getLogger,
    accuracy,
    save_checkpoint,
    load_model_parameters
)

logger = getLogger()


# DICT with number of classes for each  dataset
NUM_CLASSES = {
    'autism' :2,
}
# Create Finetune Model
class Finetune_Model(torch.nn.Module):
    def __init__(
        self,
        base_arch,
        num_ftrs=512,
        num_classes=2,
        use_dropout=False,
        use_bn=False,
        use_l2_norm=False,
        dropout=0.9
    ):
        super(Finetune_Model, self).__init__()
        self.base = base_arch
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm

        message = 'Classifier to %d classes;' % (num_classes)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_bn: message += ' + final BN'
        print(message)

        if self.use_bn:
            print("Adding BN to Classifier")
            self.final_bn = nn.BatchNorm1d(num_ftrs)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.linear_layer = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.linear_layer)
            self.classifier = torch.nn.Sequential(
                self.final_bn,
                self.linear_layer
            )
        else:
            self.classifier = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.classifier)
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
    
    def forward(self, x):
        x = self.base(x).squeeze()
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


class Finetune_Model_Agg(torch.nn.Module):
    def __init__(
        self,
        base_arch,
        pooling_arch,
        num_ftrs=512,
        num_classes=2,
        use_dropout=False,
        use_bn=False,
        use_l2_norm=False,
        dropout=0.9,
    ):
        super(Finetune_Model_Agg, self).__init__()
        self.base = base_arch
        self.pooling_arch = pooling_arch
        self.num_chunk = 2
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm


        message = 'Classifier to %d classes;' % (num_classes)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_bn: message += ' + final BN'
        print(message)

        if self.use_bn:
            print("Adding BN to Classifier")
            self.final_bn = nn.BatchNorm1d(num_ftrs)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.linear_layer = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.linear_layer)
            self.classifier = torch.nn.Sequential(
                self.final_bn,
                self.linear_layer
            )
        else:
            self.classifier = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.classifier)
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x):
        # Encode
        x = self.base(x).squeeze()

        # Pooling
        x = self.pooling_arch(x)

        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


# Load finetune model and training params
def load_model_finetune(
    args, model, num_ftrs, num_classes, agg_model=False,
    pooling_arch=None, use_dropout=False, use_bn=False,
    use_l2_norm=False, dropout=0.9,
):
    print('Using non-agg GDT model')
    new_model = Finetune_Model(
            model,
            num_ftrs,
            num_classes,
            use_dropout=use_dropout,
            use_bn=use_bn,
            use_l2_norm=use_l2_norm,
            dropout=dropout
        )
    return new_model


def main(args, writer):

    # Create Logger
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )
   
    # Set CudNN benchmark
    torch.backends.cudnn.benchmark = True
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load model
    logger.info("Loading model")
    model = load_model(
        model_type=args.model,
        vid_base_arch=args.vid_base_arch,
        vid2_base_arch=args.vid_base_arch,
        pretrained=args.pretrained,
        norm_feat=False,
        use_mlp=args.use_mlp,
        num_classes=256,
        args=args,
    )

    # Load model weights
    weight_path_type = type(args.weights_path)
    if weight_path_type == str:
        weight_path_not_none = args.weights_path != 'None'
    else:
        weight_path_not_none = args.weights_path is not None
    if not args.pretrained and weight_path_not_none:
        logger.info("Loading model weights")
        if os.path.exists(args.weights_path):
            ckpt_dict = torch.load(args.weights_path)
            try:
                model_weights = ckpt_dict["state_dict"]
            except:
                model_weights = ckpt_dict["model"]
            epoch = ckpt_dict["epoch"]
            logger.info(f"Epoch checkpoint: {epoch}")
            load_model_parameters(model, model_weights)
    logger.info(f"Loading model done")

    # Add FC layer to model for fine-tuning or feature extracting
    model = load_model_finetune(
        args,
        model.video_network.base,
        pooling_arch=model.video_pooling if args.agg_model else None,
        num_ftrs=model.encoder_dim,
        num_classes=NUM_CLASSES[args.dataset],
        use_dropout=args.use_dropout,
        use_bn=args.use_bn,
        use_l2_norm=args.use_l2_norm,
        dropout=0.9,
        agg_model=args.agg_model,
    )

    # Create DataParallel model
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    # Get params for optimization
    params = []
    if args.feature_extract: # feature_extract only classifer
        logger.info("Getting params for feature-extracting")
        for name, param in model_without_ddp.classifier.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {
                    'params': param, 
                    'lr': args.head_lr, 
                    'weight_decay': args.weight_decay
                })
    else: # finetune
        logger.info("Getting params for finetuning")
        for name, param in model_without_ddp.classifier.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {
                    'params': param, 
                    'lr': args.head_lr, 
                    'weight_decay': args.weight_decay
                })
        for name, param in model_without_ddp.base.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {   
                    'params': param, 
                    'lr': args.base_lr, 
                    'weight_decay': args.wd_base
                })
        if args.agg_model:
            logger.info("Adding pooling arch params to be optimized")
            for name, param in model_without_ddp.pooling_arch.named_parameters():
                if param.requires_grad and param.dim() >= 1:
                    logger.info(f"Adding {name}({param.shape}), wd: {args.wd_tsf}")
                    params.append(
                        {
                            'params': param, 
                            'lr': args.tsf_lr, 
                            'weight_decay': args.wd_tsf
                        })
                else:
                    logger.info(f"Not adding {name} to be optimized")


    logger.info('\n===========Check Grad============')
    for name, param in model_without_ddp.named_parameters():
        logger.info((name, param.requires_grad))
    logger.info('=================================\n')

    logger.info("Creating VV Datasets")
    dataset = VideoDataset(dataset=args.setup,split='train') 
    dataset_test = VideoDataset(dataset=args.setup,split=args.test_set)
    # Creating dataloaders
    logger.info("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    # linearly scale LR and set up optimizer
    logger.info(f"Using SGD with lr: {args.head_lr}, wd: {args.weight_decay}")
    optimizer = torch.optim.SGD(
        params,
        lr=args.head_lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    data_loader, data_loader_test, model, optimizer = accelerator.prepare(data_loader, data_loader_test, model, optimizer)
    # Multi-step LR scheduler
    if args.use_scheduler:
        milestones = [int(lr) - args.lr_warmup_epochs for lr in args.lr_milestones.split(',')]
        logger.info(f"Num. of Epochs: {args.epochs}, Milestones: {milestones}")
        if args.lr_warmup_epochs > 0:
            logger.info(f"Using scheduler with {args.lr_warmup_epochs} warmup epochs")
            scheduler_step = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=args.lr_gamma
            )
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=8,
                total_epoch=args.lr_warmup_epochs, 
                after_scheduler=scheduler_step
            )
        else: # no warmp, just multi-step
            logger.info("Using scheduler w/out warmup")
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=args.lr_gamma
            )
    else:
        lr_scheduler = None


    # Only perform evalaution
    if args.test_only:
        scores_val = evaluate(
            model, 
            device,
            data_loader_test,
            epoch=args.start_epoch, 
            writer=writer,
            ds=args.dataset,
        )
        _, vid_acc1, vid_acc5 = scores_val
        return vid_acc1, vid_acc5, args.start_epoch


    best_acc = 0
    test_arr=[]
    results ={#'train_loss': [],
              'train_acc@1':[],
               'val_loss': [], 'val_acc@1': []}
    for epoch in range(1, args.epochs+1):
        logger.info(f'Start training epoch: {epoch}')
        train_loss, acc = train(
            model,
            device,
            accelerator,
            optimizer,
            data_loader,
            epoch,
            writer=writer,
            ds=args.dataset,
        )
        #results['train_loss'].append(train_loss)
        results['train_acc@1'].append(acc)
        # eval for one epoch
        logger.info(f'Start evaluating epoch: {epoch}')
        lr_scheduler.step()
        loss, val_acc = evaluate(
                model,
                accelerator,
                device,
                data_loader_test,
                epoch=epoch,
                writer=writer,
                ds=args.dataset,
            )
        test_arr.append(val_acc)
        results['val_loss'].append(loss)
        results['val_acc@1'].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
        print('Best Accuracy in loop : {:.4f}'.format(best_acc))

    print('best accuracy: {:.4f}'.format(best_acc))
    print('Average accuracy : {:.4f}'.format(mean(test_arr)))


def train(
    model,
    device,
    accelerator,
    optimizer,
    loader,
    epoch,
    writer=None,
    ds='autism',
):
    # Put model in train mode
    model.train()

    
    # training statistics

    criterion = nn.CrossEntropyLoss().to(device)
    correct = 0
    total_num = 0.0
    total_loss =0.0
    for idx, batch in enumerate(loader):
        # measure data loading time

        # forward
        video, target, file_name = batch
        video, target = video.to(device), target.to(device)
        total_num += video.size(0)
        output = model(video)

        # compute cross entropy loss
        loss = criterion(output, target)
        pred = output.argmax(dim=1)
        correct += torch.eq(pred, target).sum().float().item()

        # compute the gradients
        optimizer.zero_grad()
        accelerator.backward(loss)
        #loss.backward()

        # step
        optimizer.step()
        total_loss += loss.item() * video.size(0)


        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.3f}\t'
                  'Acc@1: {acc:.2f}'.format(epoch, idx + 1, len(loader),loss=(total_loss / total_num), acc=((correct / total_num) *100)))
            sys.stdout.flush()
    return total_loss / total_num, correct / total_num *100



def evaluate(model, accelerator, device,val_loader, epoch=0, writer=None, ds='hmdb51'):

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    total_num ,total_loss, correct= 0.0, 0.0, 0
    names=[]
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):

            video, target, filename = batch

            # move to gpu
            video = video.to(device)
            target = target.to(device)
            bsz = target.shape[0]
            total_num += video.size(0)

            # compute output and loss
            output = model(video)
            output = accelerator.gather(output)
            target = accelerator.gather(target)
            # Example of use with a *Datasets.Metric*
            
            loss = criterion(output.view(video.size(0), -1), target)

            names.append(filename)
            # update metric
            total_loss += loss.item() * video.size(0)

            pred = output.argmax(dim=-1)
            #Prediction Saving Part
            correct += torch.eq(pred, target).sum().float().item()
            predictions = pred.data.cpu().tolist()
            labels = target.data.cpu().tolist()
           

            if idx % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f} \t'
                      'Acc@1: {acc:.3f}'.format(
                       idx,len(val_loader),
                       loss=(total_loss / total_num), acc=((correct / total_num)*100)))

    print(' * Acc@1 {:.3f}'.format((correct / total_num)*100))
    return total_loss / total_num, ((correct / total_num)*100)



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
    parser = argparse.ArgumentParser(description='Video Action Finetune')
    parser.register('type', 'bool', str2bool)

    ### DATA
    parser.add_argument('--dataset', default='autism', type=str,
                        help='name of dataset')
    parser.add_argument('--setup', default = 'modal1', type=str,
                       help='dataset on which model to be evaluated')
    parser.add_argument('--workers', default=16, type=int, 
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--test_set',type=str,default= 'test',
                         help='test_set')

    ### MODEL
    parser.add_argument('--weights-path', default='', type=str,
                        help='Path to weights file')
    parser.add_argument('--ckpt-epoch', default='0', type=str,
                        help='Epoch of model checkpoint')
    '''parser.add_argument('--model', default='av_gdt', help='model',
        choices=['av_gdt', 'vid_text_gdt', 'stica'])'''
    parser.add_argument('--vid-base-arch', default='r2plus1d_18', type=str,
                        help='Video Base Arch for A-V model',
                        choices=['r2plus1d_18', 'r2plus1d_34'])
    parser.add_argument('--pretrained', default='False', type='bool', 
                        help='Use pre-trained models or not')
    parser.add_argument('--agg-model', default='False', type='bool', 
                        help="Aggregate model with transformer")



    ### TRAINING
    
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--weight-decay', default=0.005, type=float,
                        help='weight decay for classifier')
    parser.add_argument('--wd_tsf', default=0.005, type=float,
                        help='transformer wd')

    ### LOGGING
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', type=str,
                        help='path where to save')
    

    parser.add_argument('--start-epoch', default=1, type=int, 
                        help='start epoch')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.dump_path = args.output_dir
    args.rank = 0


    # Make output dir
    tbx_path = os.path.join(args.output_dir, 'tensorboard')
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up tensorboard
    writer = writer = SummaryWriter(tbx_path)
    writer.add_text("namespace", repr(args))


    #runs the code
    main(args, writer)

