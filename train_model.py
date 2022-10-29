# Some codes refers to "Stacked cross refinement network for edge-aware salient object detection", ICCV 2019.
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from Contrast_loss import ContrastLoss
import matplotlib.pyplot as plt
import os, argparse
from datetime import datetime
from auxils.data import get_loader
from auxils.func import label_edge_prediction, AvgMeter
from model.MGuidNet import MGuid

# hyper-parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=7, help='batch size')
parser.add_argument('--trainsize', type=int, default=352, help='input size')
parser.add_argument('--trainset', type=str, default='DUTS-TR', help='training  dataset')
opt = parser.parse_args()

# data preparing
data_path = './datasets/'
image_root = data_path + opt.trainset + '/DUTS-TR-Image/'
gt_root = data_path + opt.trainset + '/DUTS-TR-Mask/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

# build models
model = MGuid()
model.cuda()
params = model.parameters()
loss_list = []
loss_record1_list = []
loss_record2_list = []
loss_record3_list = []
loss_record4_list = []
loss_record5_list = []
loss_record6_list = []
epoch_list = []
optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
CE = torch.nn.BCEWithLogitsLoss()
CEN = ContrastLoss()
size_rates = [0.75, 1, 1.25]

# training
def main():
 for epoch in range(0, opt.epoch):
    model.train()
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5, loss_record6 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            gts[gts > 0.5020] = 1
            gts[gts <= 0.5020] = 0
            # edge prediction
            gt_edges = label_edge_prediction(gts)
            # multi-scale training samples
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_edges = F.upsample(gt_edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # forward
            pred_sal, pred_edge, pred_fsal = model(images)
            #loss
            loss1 = CE(pred_sal, gts)
            loss1_2 = CEN(pred_sal, gts)
            loss2 = CE(pred_edge, gt_edges)
            loss2_2 = CEN(pred_edge, gt_edges)
            loss3 = CE(pred_fsal, gts)
            loss3_2 = CEN(pred_fsal, gts)
            if epoch<10:
                loss = loss1 + loss2 + loss3
            else:
                loss = loss1 + loss2 + loss3 + 0.1*loss1_2 + 0.1*loss2_2 + 0.1*loss3_2
            loss.backward()
            optimizer.step()
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss1_2.data, opt.batchsize)
                loss_record3.update(loss2.data, opt.batchsize)
                loss_record4.update(loss2_2.data, opt.batchsize)
                loss_record5.update(loss3.data, opt.batchsize)
                loss_record6.update(loss3_2.data, opt.batchsize)
        if i % 1000 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}, Loss5: {:.4f}, Loss6: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(), loss_record6.show()))
    scheduler.step()
    if epoch<10:
        loss_list.append(loss_record1.show() + loss_record3.show() + loss_record5.show())
        loss_record2_list.append(0.1 * loss_record2.show())
        loss_record4_list.append(0.1 * loss_record4.show())
        loss_record6_list.append(0.1 * loss_record6.show())
    else:
        loss_list.append(loss_record1.show() + loss_record3.show() + loss_record5.show() + 0.1*loss_record2.show() + 0.1*loss_record4.show() + 0.1*loss_record6.show())
        loss_record2_list.append(0.1 * loss_record2.show())
        loss_record4_list.append(0.1 * loss_record4.show())
        loss_record6_list.append(0.1 * loss_record6.show())
    loss_record3_list.append(loss_record3.show())
    loss_record1_list.append(loss_record1.show())
    loss_record5_list.append(loss_record5.show())
    epoch_list.append(epoch)
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 ax1.plot(epoch_list, loss_list, 'b', label='total loss')
 ax1.plot(epoch_list, loss_record1_list, 'c--', label='record1 loss')
 ax1.plot(epoch_list, loss_record2_list, 'c-', label='record2 loss')
 ax1.plot(epoch_list, loss_record3_list, 'm--', label='record3 loss')
 ax1.plot(epoch_list, loss_record4_list, 'm-', label='record4 loss')
 ax1.plot(epoch_list, loss_record5_list, 'g--', label='record5 loss')
 ax1.plot(epoch_list, loss_record6_list, 'g-', label='record6 loss')
 ax1.spines['right'].set_color('none')
 ax1.spines['top'].set_color('none')
 ax1.xaxis.set_ticks_position('bottom')
 ax1.yaxis.set_ticks_position('left')
 ax1.legend(loc='upper left')
 plt.savefig('loss.png')
 plt.show()
if __name__ == '__main__':
    main()
save_path = './models/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(model.state_dict(), save_path + opt.trainset + '_w.pth')
