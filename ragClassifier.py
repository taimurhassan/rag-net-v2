import cv2
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from codebase.train import customLoss
import six
from codebase.models.ragnet import *
from codebase.models.segnet import *
from codebase.models.unet import *
from codebase.models.pspnet import *
from codebase.models.config import IMAGE_ORDERING


import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

modelFile = "model.h5"
#model = model_from_json(open(modelFile).read())
#model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))
model = load_model(modelFile, compile=False)

model.summary()

doTraining = True

if doTraining == True:
    modelC = get_ragnet_classifier(input_height=576, input_width=768, model=model)
    modelC.save("modelC.h5")
else:
    modelC = load_model("modelC.h5")
    modelC.summary()

folder = "testingDataset/test_images/" # for OCT classification

# for glaucoma classification using fundus images, please uncomment the line below (please note that the '/codebase/models/trainingSet/' and '/codebase/models/validationSet/' should be updated accordingly for glaucomic classification through fundus images)
#folder = "testingDataset/fundus_test_images/" 

for filename in os.listdir(folder):
    inp=os.path.join(folder,filename)
    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, 1)
        inp = inp[:,:,::-1]
#        plt.imshow(inp)
#        plt.show()

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    width = orininal_w
    height = orininal_h
    
    imgNorm = "sub_mean"
    
    if type(inp) is np.ndarray:
        img = inp
        
    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if IMAGE_ORDERING == 'channels_first':
        img = np.rollaxis(img, 2, 0)
        
    predictions = modelC.predict(np.array([img]))
    decision = np.argmax(predictions, axis = 1)
    print(decision) 
    out_fname=os.path.join("testingDataset/diagnosisResults/",filename)
    sio.savemat(out_fname[:-4]+'.mat', {'decision': decision, 'imageName': filename})