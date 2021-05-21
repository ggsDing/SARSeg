import os
import time
import random
import numpy as np
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))

###############################################
from datasets import RS_XT as RS
#from models.MS_FCN import MS_FCN as Net
from models.MPResNet import MPResNet as Net
NET_NAME = 'MPResNet'
DATA_NAME = 'XT'
###############################################

from utils.loss import CrossEntropyLoss2d, FocalLoss2d
from utils.utils import accuracy, FWIoU, intersectionAndUnion, AverageMeter
    
args = {
    'lr': 0.1,
    'gpu': True,
    'epochs': 200,
    'momentum': 0.9,
    'print_freq': 10,
    'predict_step': 5,
    'val_batch_size': 8,
    'train_batch_size': 8,
    'weight_decay': 5e-4,
    'lr_decay_power': 1.5,
    'train_crop_size': False,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME)
}

writer = SummaryWriter(args['log_dir'])
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])

def main():
    net = Net(4, num_classes=RS.num_classes+1) #, pretrained=True
    if args['gpu']: net = net.cuda()
        
    train_set = RS.PolSAR(mode='train', random_flip=True, crop_size=args['train_crop_size'])
    val_set = RS.PolSAR(mode='val', random_flip=False, crop_size=False)

    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0)
    #criterion = FocalLoss2d(gamma=2.0, ignore_index=0)
    if args['gpu']: criterion = criterion.cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader, net, criterion, optimizer, scheduler, args, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, scheduler, args, val_loader):
    bestaccT=0
    bestfwiou=0.5
    bestaccV=0.0
    bestloss=1
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])
    curr_epoch=0
    while True:
        if args['gpu']: torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        
        curr_iter = curr_epoch*len(train_loader)
        for i, (imgs, labels) in enumerate(train_loader):
            running_iter = curr_iter+i+1
            adjust_lr(optimizer, running_iter, all_iters)
            #imgs = torch.squeeze(imgs)
            imgs = imgs.float()
            labels = labels.long()
            #imgs, labels = data
            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.cuda().long()

            optimizer.zero_grad()
            outputs, aux = net(imgs) #
            assert outputs.shape[1] == RS.num_classes+1
            loss_main = criterion(outputs, labels)
            loss_aux = criterion(aux, labels)
            loss = loss_main*0.7 + loss_aux*0.3
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            _, preds = torch.max(outputs, dim=1)
            preds = preds.numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, _ = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_main_loss.update(loss.cpu().detach().numpy())
            train_aux_loss.update(loss_aux.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_main_loss.val, acc_meter.val*100))
                writer.add_scalar('train_main_loss', train_main_loss.val, running_iter)
                writer.add_scalar('train_accuracy', acc_meter.val, running_iter)
                writer.add_scalar('train_aux_loss', train_aux_loss.avg, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)
                    
        acc_v, fwiou_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT=acc_meter.avg
        
        if fwiou_v>bestfwiou:
            bestfwiou=fwiou_v
            bestloss=loss_v
            bestaccV=acc_v
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%de_OA%.2f_fwIoU%.2f.pth'%(curr_epoch, acc_v*100, fwiou_v*100)) )
        print('Total time: %.1fs Best rec: Train acc %.2f, Val acc %.2f fwiou %.2f, Val_loss %.4f' %(time.time()-begin_time, bestaccT*100, bestaccV*100, bestfwiou*100, bestloss))
        curr_epoch += 1
        #scheduler.step()
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    if args['gpu']:
        torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()
    fwIoU_meter = AverageMeter()

    for vi, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.float()
        labels = labels.long()

        if args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()

        with torch.no_grad():
            outputs, aux = net(imgs)
            loss = criterion(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        _, preds = torch.max(outputs, dim=1)
        preds = preds.numpy()
        for (pred, label) in zip(preds, labels):
            acc, valid_sum = accuracy(pred, label)
            fwiou = FWIoU(pred.squeeze(), label.squeeze(), ignore_zero=True)
            acc_meter.update(acc)
            fwIoU_meter.update(fwiou)

        if curr_epoch%args['predict_step']==0 and vi==0:
            pred_color = RS.Index2Color(preds[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f, Accuracy: %.2f, fwIoU: %.2f'%(curr_time, val_loss.average(), acc_meter.average()*100, fwIoU_meter.average()*100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)
    writer.add_scalar('val_fwIoU', fwIoU_meter.average(), curr_epoch)

    return acc_meter.avg, fwIoU_meter.avg, val_loss.avg

def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
        
if __name__ == '__main__':
    main()
