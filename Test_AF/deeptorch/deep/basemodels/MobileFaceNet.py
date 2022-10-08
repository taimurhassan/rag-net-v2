# import sys
# sys.path.append('C:/Users/mohda/Downloads/DeepFace PyTorch/deep/basemodels/')
import os
import torch
from deep.basemodels.QuantFace.backbones.mobilefacenet import MobileFaceNet


def loadModel(eval=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    output_folder = "/Test_AF/deeptorch/deep/basemodels/QuantFace/weights/181952backbone.pth"
    gpu_id = 0
    
    # type of network to train [iresnet100 | iresnet50| iresnet18]
    network = "mobilefacenet"
    embedding_size = 128

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #This way
    # backbone = mobilefacenet.MobileFaceNet(input_size=(112,112), embedding_size=embedding_size, output_name="GDC", attention="none").to(device)
    #OR           

    backbone = MobileFaceNet(input_size=(112,112), embedding_size=embedding_size, output_name="GDC", attention="none").to(device).to(device)
    backbone.load_state_dict(torch.load(output_folder))
    model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
            
    if eval:
        model.eval()
        model.to(device)
    return model

def build_model(name="mobilefacenet"):

    embedding_size = 128
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if name == "mobilefacenet":
        backbone = MobileFaceNet(input_size=(112,112), embedding_size=embedding_size, output_name="GDC", attention="none").to(device)

    return backbone

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