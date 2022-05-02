import cv2
import os
import torch
import time
import sys
from torch.utils.data import DataLoader
import csv
import numpy as np
import pandas as pd
from networks.vision_transformer_multi_scale_class import SwinUnet as ViT_seg
from config import get_config
from datasets.create_dataset import Mydataset,for_train_transform,test_transform
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_test_path', '-ivt', type=str, help='imgs val data path.')
parser.add_argument('--labels_test_path', '-lvt', type=str, help='labels val data path.')

parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
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
parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()
# print('model_ok')

res = args.resize

test_transform = test_transform


test_imgs_dir = args.imgs_test_path
test_csv = args.labels_test_path
df_test = pd.read_csv(test_csv)#[0:50]
all_test_imgs = [''.join([test_imgs_dir,'/',i,'.jpg']) for i in df_test['image_id']]
all_test_masks = [''.join([str(i)]) for i in df_test['label']]
test_masks = all_test_masks
test_imgs = [cv2.resize(cv2.imread(i), (args.resize,args.resize))[:,:,::-1] for i in all_test_imgs]
test_ds = Mydataset(test_imgs, test_masks, test_transform)
test_dl = DataLoader(test_ds,batch_size=1,pin_memory=False,num_workers=4,)

best_acc_final = []
from sklearn import metrics
def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    f1_score = metrics.f1_score(label, binary_score, average='macro')
    precision = metrics.precision_score(label, binary_score)
    recall = metrics.recall_score(label, binary_score, average='macro')
    jaccard = metrics.jaccard_score(label, binary_score, average='macro')
    CM = metrics.confusion_matrix(label, binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec



def main():

    config = get_config(args)
    model = ViT_seg(config, img_size=args.resize, num_classes=args.num_classes).to('cuda')
    model.load_state_dict(torch.load('checkpoint/classificationlr0.0001bs16/ckpt.pth'))
    model.eval()

    correct_val = 0
    total_val = 0
    number=0

    pred_score = []
    label_val = []
    metrics_for_csv = []
    with torch.no_grad():
        for batch_idx,(imgs, masks) in enumerate(test_dl):
            number+=1
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            masks_pred = model(imgs)
            correct_val += (torch.max(masks_pred, 1)[1].view(masks_cuda.size()).data == masks_cuda.data).sum()
            total_val += test_dl.batch_size
            test_acc = ((correct_val.item()) / total_val)
            sys.stdout.write('\r%d/%d/%s' % (number,correct_val.item(), len(test_dl)))
            pred_score.append(torch.softmax(masks_pred[0], dim=0).cpu().data.numpy())
            label_val.append(masks_cuda[0].cpu().data.numpy())
    print('test_acc',test_acc)
    num=3
    pro_score = np.array(pred_score)
    label_val = np.array(label_val)
    pro_score_all = np.array(pro_score)
    binary_score_all = np.eye(num)[np.argmax(np.array(pro_score), axis=-1)]
    label_val_all = np.eye(num)[np.int64(np.array(label_val))]

    if num == 3:
        metrics_for_csv.append(['0', '1','2'])

    metrics_for_csv.append(['acc', 'AP', 'auc', 'f1_score', 'precision', 'recall','jaccard', 'sens', 'spec'])
    for i in range(num):
        label_val_cls0 = label_val_all[:, i-1]
        pred_prob_cls0 = pro_score_all[:, i-1]
        binary_score_cls0 = binary_score_all[:, i-1]
        acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec = cla_evaluate(label_val_cls0,
                                                                                      binary_score_cls0,
                                                                                      pred_prob_cls0)

        line_test_cls0 = "test:acc=%f,AP=%f,auc=%f,f1_score=%f,precision=%f,recall=%f,sens=%f,spec=%f\n" % (
            acc, AP, auc, f1_score, precision, recall, sens, spec)
        print(line_test_cls0)
        metrics_for_csv.append([acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec])
    results_file = open('classification_result.csv', 'w', newline='')
    csv_writer = csv.writer(results_file, dialect='excel')
    for row in metrics_for_csv:
        csv_writer.writerow(row)


if __name__ == '__main__':
    main()

after_net_time = time.time()
print('best_acc_final_val',best_acc_final)
