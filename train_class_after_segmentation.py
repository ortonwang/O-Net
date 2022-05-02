import cv2
import os
import torch
import copy
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from fit import set_seed,write_options ,fit
from datasets.create_dataset import Mydataset,for_train_transform,test_transform
from networks.vision_transformer_multi_scale_class import SwinUnet as ViT_seg
from config import get_config


import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str, help='imgs train data path.')
parser.add_argument('--labels_train_path', '-lt', type=str, help='labels train data path.')
parser.add_argument('--imgs_val_path', '-iv', type=str, help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str, help='labels val data path.')
parser.add_argument('--imgs_test_path', '-ivt', type=str, help='imgs val data path.')
parser.add_argument('--labels_test_path', '-lvt', type=str, help='labels val data path.')

parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=8,type=int,help='batchsize')
parser.add_argument('--workers', default=16,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--warm_epoch', '-w', default=0, type=int, help='end epoch')
parser.add_argument('--end_epoch', '-e', default=20, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default= 'classification', help='checkpoint path')
# parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default='0', type=int, help='seed_num')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()
# print('model_ok')
set_seed(seed=2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = args.device
model_savedir = args.checkpoint + args.save_name +'lr'+ str(args.lr)+ 'bs'+str(args.batch_size)+'/'
save_name =model_savedir +'ckpt'
print(model_savedir)
if not os.path.exists(model_savedir):
    os.mkdir(model_savedir)
epochs = args.warm_epoch + args.end_epoch

res = args.resize
warmup_epochs,train_epochs = args.warm_epoch,args.end_epoch


epochs = args.warm_epoch + args.end_epoch

train_imgs,val_imgs = args.imgs_train_path,args.imgs_val_path
train_csv,val_csv = args.labels_train_path,args.labels_val_path

df_train = pd.read_csv(train_csv)#[0:200]
df_val = pd.read_csv(val_csv)#[0:50]

print('Start loading data.')
all_train_imgs = [''.join([train_imgs,'/',i,'.jpg']) for i in df_train['image_id']]
all_train_masks = [''.join([str(i)]) for i in df_train['label']]
all_val_imgs = [''.join([val_imgs,'/',i,'.jpg']) for i in df_val['image_id']]
all_val_masks = [''.join([str(i)]) for i in df_val['label']]
train_imgs = [cv2.resize(cv2.imread(i), (args.resize,args.resize))[:,:,::-1] for i in all_train_imgs]
train_masks = all_train_masks
val_imgs = [cv2.resize(cv2.imread(i), (args.resize,args.resize))[:,:,::-1] for i in all_val_imgs]
val_masks = all_val_masks

train_transform = for_train_transform()
test_transform = test_transform

after_read_date = time.time()
print('data_time',after_read_date-begin_time)


best_acc_final = []
def main():
    config = get_config(args)
    model = ViT_seg(config, img_size=args.resize, num_classes=args.num_classes)
    swin_multi_scale_dir = 'checkpoint/CNN_with_Swin_E_Swin_D/epoch_249.pth'  #loading segmentation weights
    model = model.to('cuda')
    model.load_from2(swin_multi_scale_dir)


    model = torch.nn.DataParallel(model)
    train_ds = Mydataset(train_imgs, train_masks, train_transform)
    val_ds = Mydataset(val_imgs, val_masks, test_transform)

    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #exp_lr_schedular = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-10)
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=(epochs//warmup_epochs))
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=(epochs//warmup_epochs))
    train_dl = DataLoader(train_ds,shuffle=True,batch_size=args.batch_size,pin_memory=False,num_workers=8,drop_last=True,)
        #prefetch_factor=4)
    val_dl = DataLoader(val_ds,batch_size=args.batch_size,pin_memory=False,num_workers=8,)
        #prefetch_factor=4)

    best_acc = 0

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(epochs):
        if epoch == warmup_epochs:
            for para in model.parameters():
                para.requires_grad = True
        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,model,train_dl,val_dl,criterion,optimizer,epochs,CosineLR)

        f = open(model_savedir + 'log'+'.txt', "a")
        f.write('epoch' + str(float(epoch)) +
                '  _train_loss'+ str(epoch_loss)+'  _val_loss'+str(epoch_test_loss)+
                ' _epoch_acc'+str(epoch_acc)+' _val_iou'+str(epoch_test_acc)+   '\n')

        if epoch_test_acc > best_acc:
            f.write( '\n' + 'here' + '\n')
            print('here')
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = epoch_test_acc
            torch.save(best_model_wts, ''.join([save_name, str(epoch), '.pth']))
        if epoch%5 ==0:
            best_model_wts2 = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts2, ''.join([save_name, str(epoch)+ '.pth']))
        f.close()
        f.close()
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)


    best_acc_final.append(best_acc)
    fig = plt.figure(figsize=(22,8))
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='valid loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(''.join([model_savedir, '_Loss.png']), bbox_inches = 'tight')

    fig = plt.figure(figsize=(22,8))
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='valid acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(''.join([model_savedir, '_Acc.png']), bbox_inches = 'tight')
    model.load_state_dict(torch.load(''.join([save_name,  '.pth'])))
    model.eval()
    correct_val = 0
    total_val = 0


    test_imgs_dir = args.imgs_test_path
    test_csv = args.labels_test_path
    df_test = pd.read_csv(test_csv)#[0:50]
    all_test_imgs = [''.join([test_imgs_dir,'/',i,'.jpg']) for i in df_test['image_id']]
    all_test_masks = [''.join([str(i)]) for i in df_test['label']]
    test_masks = all_test_masks
    test_imgs = [cv2.resize(cv2.imread(i), (args.resize,args.resize))[:,:,::-1] for i in all_test_imgs]
    test_ds = Mydataset(test_imgs, test_masks, test_transform)
    test_dl = DataLoader(test_ds,batch_size=1,pin_memory=False,num_workers=4,)

    with torch.no_grad():
        for batch_idx,(imgs, masks) in enumerate(test_dl):
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            masks_pred = model(imgs)
            # print(masks_pred,masks_cuda)
            correct_val += (torch.max(masks_pred, 1)[1].view(masks_cuda.size()).data == masks_cuda.data).sum()
            total_val += test_dl.batch_size
            test_acc = ((correct_val.item()) / total_val)
            # loss = criterion(masks_pred, masks_cuda.squeeze())
    print('test_acc',test_acc)
    write_options(model_savedir,args,best_acc_final,test_acc)


if __name__ == '__main__':
    main()

after_net_time = time.time()
print('net_time',after_net_time-after_read_date)
print('best_acc_final_val',best_acc_final)
print(save_name)
