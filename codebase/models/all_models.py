from . import ragnet

model_from_name = {}

model_from_name["ragnet_classifier"] = ragnet.get_ragnet_classifier
model_from_name["ragnet_segmentor"] = ragnet.ragnet