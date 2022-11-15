import argparse
import os

import mxnet as mx
import torch

import deep.basemodels.mixfacenets.backbones.mixnetm as mx
# import mixfacenets.backbones.mixnetm as mx

def loadModel(eval=True, net='MixFaceNetXS'):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if net == 'MixFaceNetXS':
        print('MixFaceNetXS is chosen')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/MixFaceNet-XS/295672backbone.pth'
        width_scale = 0.5
        gdw_size = 512
        embedding_size = 512
        backbone = mx.mixnet_s(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=False).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)
    
    elif net == 'ShuffleMixFaceNetXS':
        print('ShuffleMixFaceNetXS is chosen')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/ShuffleMixFaceNet-XS/216068backbone.pth'
        width_scale = 0.5
        gdw_size = 512
        embedding_size = 512
        backbone = mx.mixnet_s(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=True).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)

    elif net == 'MixFaceNetS':
        print('MixFaceNetS is chosen')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/MixFaceNet-S/295672backbone.pth'
        width_scale = 1
        gdw_size = 1024
        embedding_size = 512
        backbone = mx.mixnet_s(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=False).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)

    elif net == 'ShuffleMixFaceNetS':
        print('ShuffleMixFaceNetS is chosen')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/ShuffleMixFaceNet-S/295672backbone.pth'
        width_scale = 1
        gdw_size = 1024
        embedding_size = 512
        backbone = mx.mixnet_s(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=True).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)

    elif net == 'MixFaceNetM':
        print('MixFaceNetM is chosen')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/MixFaceNet-M/272928backbone.pth'
        width_scale = 1
        gdw_size = 1024
        embedding_size = 512
        backbone = mx.mixnet_m(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=False).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)

    elif net == 'ShuffleMixFaceNetM':
        print('ShuffleMixFaceNetM is chosen')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/ShuffleMixFaceNet-M/284300backbone.pth'
        width_scale = 1
        gdw_size = 1024
        embedding_size = 512
        backbone = mx.mixnet_m(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=True).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)

    else:
        print('MixFaceNetXS is chosen as default')
        output_folder = '/Test_AF/deeptorch/deep/basemodels/mixfacenets/weights/MixFaceNet-XS/295672backbone.pth'
        width_scale = 0.5
        gdw_size = 512
        embedding_size = 512
        backbone = mx.mixnet_s(embedding_size=embedding_size, width_scale=width_scale, gdw_size=gdw_size, shuffle=False).to( "cuda:0")
        backbone.load_state_dict(torch.load(output_folder))
        model = torch.nn.DataParallel(backbone)

    
    if eval:
        model.eval()
        model.to(device)
    return model



def inference(model, img):
    import cv2, numpy as np, torch

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5).to(device)
    # net = get_model(name, fp16=False)
    # net.load_state_dict(torch.load(weight))
    model.eval().to(device)
    # feat = net(img).numpy()
    feat = model(img)[0].tolist()
    return feat