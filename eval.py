import os
import argparse
import numpy as np
import torch.autograd
from skimage import io
from torch.utils.data import DataLoader
#################################
from scipy import stats
from datasets import RS_XT as RS
#from models.FCN_8s import FCN_res34 as Net
from models.MPResNet import MPResNet as Net
NET_NAME = 'MPResNet_test'
DATA_NAME = 'XT'
#################################

from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, FWIoU, intersectionAndUnion, AverageMeter, CaclTP

working_path = os.path.abspath('.')
args = {
    'gpu': True,
    'val_batch_size': 1,
    'output_dir': os.path.join(RS.data_dir, 'pred', NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'MPResNet_2e_OA72.41_fwIoU60.28.pth')
}
if not os.path.exists(args['output_dir']): os.makedirs(args['output_dir'])

def main():
    net = Net(4, num_classes=RS.num_classes+1)    
    net.load_state_dict(torch.load(args['load_path']), strict = False)#, strict = False
    net = net.cuda()
    net.eval()
    print('Model loaded.')
    pred_path = os.path.join(RS.data_dir, 'pred', NET_NAME)
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')
    
    val_set = RS.PolSAR('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    predict(net, val_set, val_loader, pred_path, f)
    f.close()

def predict(net, pred_set, pred_loader, pred_path, f_out=None):
    output_info = f_out is not None

    acc_meter = AverageMeter()
    fwIoU_meter = AverageMeter()
    TP_meter = AverageMeter()
    pred_meter = AverageMeter()
    label_meter = AverageMeter()

    for vi, data in enumerate(pred_loader):
        with torch.no_grad():
            img, label = data
            if args['gpu']:
                img = img.cuda().float()
                label = label.cuda().float()

            output, aux = net(img)#
            
        output = output.detach().cpu()
        _, pred = torch.max(output, dim=1)
        pred = pred.squeeze(0).numpy()
        
        label = label.detach().cpu().numpy()
        acc, _ = accuracy(pred, label)
        fwiou = FWIoU(pred.squeeze(), label.squeeze(), ignore_zero=True)
        
        acc_meter.update(acc)
        fwIoU_meter.update(fwiou)
        pred_color = RS.Index2Color(pred)
        mask_name = pred_set.get_mask_name(vi)
        pred_name = os.path.join(args['output_dir'], mask_name)
        io.imsave(pred_name, pred_color)
        TP, pred_hist, label_hist = CaclTP(pred, label, RS.num_classes)
        TP_meter.update(TP)
        pred_meter.update(pred_hist)
        label_meter.update(label_hist)
        print('Eval num %d/%d, Acc %.2f, fwIoU %.2f'%(vi, len(pred_loader), acc*100, fwiou*100))       
        if output_info:
            f_out.write('Eval num %d/%d, Acc %.2f, fwIoU %.2f\n'%(vi, len(pred_loader), acc*100, fwiou*100))   

    precision = TP_meter.sum / (label_meter.sum + 1e-10) + 1e-10
    recall = TP_meter.sum / (pred_meter.sum + 1e-10) + 1e-10
    F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    F1 = np.array(F1)
    
    print(output.shape)
    print('Acc %.2f, fwIoU %.2f'%(acc_meter.avg*100, fwIoU_meter.avg*100))
    print(np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    avg_F = np.array(F1[:-1]).mean()
    print('Avg F1 %.2f'%(avg_F*100))
    if output_info:
        f_out.write('Acc %.2f\n'%(acc_meter.avg*100))
        f_out.write('Avg F1 %.2f\n'%(avg_F*100))
        f_out.write('fwIoU %.2f\n'%(fwIoU_meter.avg*100))
        f_out.write(np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    return avg_F


if __name__ == '__main__':
    main()
