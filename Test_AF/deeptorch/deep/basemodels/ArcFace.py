from deep.basemodels.arcface_torch.backbones import get_model
import torch

def loadModel(weight='C:/Users/mohda/Downloads/DeepFace PyTorch/deep/basemodels/arcface_torch/weights/ms1mv3_arcface_r18_fp16/backbone.pth', name='r18', eval=True):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = get_model(name, fp16=False)
	model.load_state_dict(torch.load(weight))
	if eval == True:
		model.eval()
		model.to(device)
	return model