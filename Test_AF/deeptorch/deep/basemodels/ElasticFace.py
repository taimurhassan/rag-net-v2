# import sys
# sys.path.append('C:/Users/mohda/Downloads/DeepFace PyTorch/deep/basemodels/')
import os
import torch
from deep.basemodels.Elasticface.backbones.iresnet import iresnet100, iresnet50


def loadModel(eval=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    output_folder = "/Test_AF/deeptorch/deep/basemodels/Elasticface/weights/ElasticFace-Arc+" # train model output folder
    gpu_id = 0
    
    # type of network to train [iresnet100 | iresnet50]
    network = "iresnet100"
    loss="ElasticArcFacePlus"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus
    s = 64.0
    m = 0.50
    std = 0.0175
    embedding_size = 512 # embedding size of model

    weights=os.listdir(output_folder)

    for w in weights:
        if "backbone" in w:
            backbone = iresnet100(num_features=embedding_size).to(f"cuda:{gpu_id}")
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