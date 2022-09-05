import argparse
import os

import mxnet as mx
import torch

import deep.basemodels.mixfacenets.backbones.mixnetm as mx
# import mixfacenets.backbones.mixnetm as mx

def loadModel(eval=True):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_folder = 'C:/Users/mohda/Downloads/DeepFace PyTorch/deep/basemodels/mixfacenets/weights/MixFaceNet-XS'

    #net paramerters
    net_name="mixfacenet"
    net_size="s"

    scale = 0.5
    gdw_size = 512
    embedding_size = 512

    weights=os.listdir(output_folder)

    for w in weights:
        if "backbone" in w:
            backbone = mx.mixnet_s(embedding_size=embedding_size, width_scale=scale, gdw_size=gdw_size).to( "cuda:0")
            backbone.load_state_dict(torch.load(os.path.join(output_folder,w)))
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