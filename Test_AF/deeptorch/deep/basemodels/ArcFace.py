from deep.basemodels.arcface_torch.backbones import get_model
import torch

def loadModel(weight='/Test_AF/deeptorch/deep/basemodels/arcface_torch/weights/ms1mv3_arcface_r100_fp16/backbone.pth', name='r100', eval=True, wei='backbone.pth'):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = get_model(name, fp16=False)
	# model.load_state_dict(torch.load(weight))
	model.load_state_dict(torch.load(wei))
	if eval == True:
		model.eval()
		model.to(device)
	return model
