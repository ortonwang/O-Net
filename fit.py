import os
import random
import torch
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def write_options(model_savedir,args,best_acc_val,test_acc):
    aaa = []
    aaa.append(['lr',str(args.lr)])
    aaa.append(['batch',args.batch_size])
    aaa.append(['save_name',args.save_name])
    aaa.append(['seed',args.batch_size])
    aaa.append(['best_val_acc',str(best_acc_val)])
    aaa.append(['warm_epoch',args.warm_epoch])
    aaa.append(['end_epoch',args.end_epoch])
    aaa.append(['test_acc',str(test_acc)])
    f = open(model_savedir+'option'+'.txt', "a")
    for option_things in aaa:
        f.write(str(option_things)+'\n')
    f.close()

def set_seed(seed=1): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def fit(epoch, model, trainloader, testloader,criterion,optimizer,epochs,CosineLR):
    with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
        scaler = GradScaler()
        if torch.cuda.is_available():
            model.to('cuda')
        running_loss = 0
        model.train()
        train_pa_whole = 0
        correct = 0
        total = 0
        for  batch_idx, (imgs, masks) in enumerate(trainloader):
            t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            with autocast():
                masks_pred = model(imgs)
                loss = criterion(masks_pred, masks_cuda.squeeze())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                correct += (torch.max(masks_pred, 1)[1].view(masks_cuda.size()).data == masks_cuda.data).sum()
                total += trainloader.batch_size
                running_loss += loss.item()
            epoch_acc = 100*((correct.item()) / total)
            t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
                          train_pa='{:.2f}%'.format(epoch_acc))
            t.update(1)
        # epoch_acc = correct / total
        epoch_loss = running_loss / len(trainloader.dataset)
    with tqdm(total=len(testloader), ncols=120, ascii=True) as t:
        test_running_loss = 0
        val_pa_whole = 0
        correct_val = 0
        total_val = 0

        model.eval()
        with torch.no_grad():
            for batch_idx,(imgs, masks) in enumerate(testloader):
                t.set_description("val(Epoch{}/{})".format(epoch, epochs))
                imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
                imgs = imgs.float()
                masks_pred = model(imgs)
                correct_val += (torch.max(masks_pred, 1)[1].view(masks_cuda.size()).data == masks_cuda.data).sum()
                total_val += testloader.batch_size
                epoch_test_acc = 100*((correct_val.item()) / total_val)
                loss = criterion(masks_pred, masks_cuda.squeeze())
                test_running_loss += loss.item()

                t.set_postfix(loss='{:.3f}'.format(test_running_loss / (batch_idx + 1)),
                              val_pa='{:.2f}%'.format(epoch_test_acc))
                t.update(1)
        # epoch_test_acc = test_correct / test_total
        epoch_test_loss = test_running_loss / len(testloader.dataset)
        #if epoch > 2:
        CosineLR.step()
        return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
