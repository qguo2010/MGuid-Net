import torch
import torch.nn.functional as F


class ContrastLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, y):
        x = self.sig(x)
        self.mean = []
        self.std = []
        self.num = []
        self.loss = []
        area_f = torch.where(y > 0.5020)
        area_feature = x[area_f]
        self.num.append(len(area_feature))
        self.mean.append(area_feature.mean())
        loss_t1 = ((area_feature - 1) ** 2).sum() / (self.num[0] - 1)
        loss_t1 = self.sig(loss_t1)
        self.std.append(loss_t1)
        loss1 = (-torch.log(1 - loss_t1))
        area_b = torch.where(y <= 0.5020)
        area_back = x[area_b]
        self.num.append(len(area_back))
        self.mean.append(area_back.mean())
        loss_t2 = ((area_back - 0) ** 2).sum() / (self.num[1] - 1)
        loss_t2 = self.sig(loss_t2)
        self.std.append(loss_t2)
        loss2 = (-torch.log(1 - loss_t2))
        l_c = loss1 + 2*loss2

        return l_c