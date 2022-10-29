import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import imageio
from auxils.data import test_dataset
from model.ResNet_models import MGuid

model = MGuid()
model.load_state_dict(torch.load('./models/DUTS-TR_W.pth'))
model.cuda()
model.eval()

data_path = './'
valset = ['DUTS-TE']
for dataset in valset:
    save_path = './saliency_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = data_path + dataset + '/Image/'
    gt_root = data_path + dataset + '/Mask/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max() + 1e-8)
            image = Variable(image).cuda()
            res, edge, fres = model(image)
            fres = F.upsample(fres, size=gt.shape, mode='bilinear', align_corners=True)
            fres = fres.sigmoid().data.cpu().numpy().squeeze()
            fres = 255 * fres
            fres = fres.astype(np.uint8)
            imageio.imwrite(save_path + name + '.png', fres)

