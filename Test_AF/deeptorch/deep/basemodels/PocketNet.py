import torch
from torch.autograd import Variable
import numpy as np
import os


from deep.basemodels.Pocketnet.backbones.iresnet import iresnet100
from deep.basemodels.Pocketnet.backbones.augment_cnn import AugmentCNN
from deep.basemodels.Pocketnet.backbones import genotypes as gt


# dataset = "emoreKD"
# embedding_size = 128
# output_folder = "PocketNet/weights/PocketNetS-128"
# scale=1.0
# global_step=0
# s=64.0
# m=0.5

# channel=16
# n_layers=18
# gpu_id = 0

# genotypes = dict({
#         "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
#         "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })

# genotype = gt.from_str(genotypes["softmax_cifar10"])


def loadModel(eval=True):
    
    output_folder = "C:/Users/mohda/Downloads/DeepFace PyTorch/deep/basemodels/Pocketnet/weights/PocketNetS-128"

    embedding_size = 128
    channel=16
    n_layers=18
    gpu_id = 0

    genotypes = dict({
        "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
        "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })

    genotype = gt.from_str(genotypes["softmax_casia"])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    backbone = AugmentCNN(C=channel, n_layers=n_layers, genotype=genotype, stem_multiplier=4,
                       emb=embedding_size).to(device)
                       
    weights=os.listdir(output_folder)

    for w in weights:
        if "backbone" in w:
            backbone=AugmentCNN(C=channel, n_layers=n_layers, genotype=genotype, stem_multiplier=4,
                    emb=embedding_size).to(f"cuda:{gpu_id}")
            backbone.load_state_dict(torch.load(os.path.join(output_folder,w)))
            model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
        
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