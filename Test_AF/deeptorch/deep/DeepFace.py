import warnings

warnings.filterwarnings("ignore")

# import sys
# sys.path.append("C:/Users/mohda/Downloads/DeepFace PyTorch/")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

#---------------------------------
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

#---------------------------------
# from basemodels import ShuffleFaceNet, FaceNet, MobileFaceNet
# from deepface.basemodels import ArcFace, Boosting, VGGFace
# from deep.basemodels import ShuffleFaceNet, FaceNet, MobileFaceNet, ArcFace, PocketNet, ElasticFace, MixFaceNetXS, MixFaceNetM, Boosting, VGGFace
from deep.basemodels import PocketNet, Boosting, ShuffleFaceNet
from deep.commons import functions, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)


def build_model(model_name):

	"""
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition

	Returns:
		built deepface model
	"""

	global model_obj #singleton design pattern

	models = {
		# 'VGG-Face': VGGFace.loadModel,
		'ShuffleFaceNet': ShuffleFaceNet.loadModel,
		# 'MobileFaceNet': MobileFaceNet.loadModel,
		# 'FaceNet': FaceNet.loadModel,
		# 'ArcFace': ArcFace.loadModel,
		'PocketNet': PocketNet.loadModel
		# 'ElasticFace': ElasticFace.loadModel,
		# 'MixFaceNetXS': MixFaceNetXS.loadModel,
		# 'MixFaceNetM': MixFaceNetM.loadModel,

	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model()
			model_obj[model_name] = model
			#print(model_name," built")
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]

def verify(img1_path, img2_path = '', model_name = 'PacketNet', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):

	"""
	This function verifies an image pair is same person or different persons.

	Parameters:
		img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

		e.g. img1_path = [
			['img1.jpg', 'img2.jpg'],
			['img2.jpg', 'img3.jpg']
		]

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble

		distance_metric (string): cosine, euclidean, euclidean_l2

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

			model = DeepFace.build_model('VGG-Face')

		enforce_detection (boolean): If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

		prog_bar (boolean): enable/disable a progress bar

	Returns:
		Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

		{
			"verified": True
			, "distance": 0.2563
			, "max_threshold_to_verify": 0.40
			, "model": "VGG-Face"
			, "similarity_metric": "cosine"
		}

	"""

	tic = time.time()

	img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)

	resp_objects = []

	#--------------------------------

	if model_name == 'Ensemble':
		# model_names = ['ShuffleFaceNet', 'MobileFaceNet', 'FaceNet', 'ArcFace', 'PocketNet', 'ElasticFace', 'MixFaceNetXS', 'MixFaceNetM']
		model_names = ['PacketNet']
		metrics = ["cosine", "euclidean", "euclidean_l2"]
	else:
		model_names = []; metrics = []
		model_names.append(model_name)
		metrics.append(distance_metric)

	#--------------------------------

	if model == None:
		if model_name == 'Ensemble':
			models = Boosting.loadModel()
		else:
			model = build_model(model_name)
			models = {}
			models[model_name] = model
	else:
		if model_name == 'Ensemble':
			Boosting.validate_model(model)
			models = model.copy()
		else:
			models = {}
			models[model_name] = model

	#------------------------------

	disable_option = (False if len(img_list) > 1 else True) or not prog_bar

	pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = disable_option)

	for index in pbar:

		instance = img_list[index]

		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]; img2_path = instance[1]

			ensemble_features = []

			for i in  model_names:
				custom_model = models[i]

				#img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'mtcnn'
				img1_representation = represent(img_path = img1_path
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)

				img2_representation = represent(img_path = img2_path
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)

				#----------------------
				#find distances between embeddings

				for j in metrics:

					if j == 'cosine':
						distance = dst.findCosineDistance(img1_representation, img2_representation)
					elif j == 'euclidean':
						distance = dst.findEuclideanDistance(img1_representation, img2_representation)
					elif j == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
					else:
						raise ValueError("Invalid distance_metric passed - ", distance_metric)

					distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
					#----------------------
					#decision

					if model_name != 'Ensemble':

						threshold = dst.findThreshold(i, j)

						if distance <= threshold:
							identified = True
						else:
							identified = False

						resp_obj = {
							"verified": identified
							, "distance": distance
							, "threshold": threshold
							, "model": model_name
							, "detector_backend": detector_backend
							, "similarity_metric": distance_metric
						}

						if bulkProcess == True:
							resp_objects.append(resp_obj)
						else:
							return resp_obj

					else: #Ensemble

						#this returns same with OpenFace - euclidean_l2
						if i == 'OpenFace' and j == 'euclidean':
							continue
						else:
							ensemble_features.append(distance)

			#----------------------

			if model_name == 'Ensemble':

				boosted_tree = Boosting.build_gbm()

				prediction = boosted_tree.predict(np.expand_dims(np.array(ensemble_features), axis=0))[0]

				verified = np.argmax(prediction) == 1
				score = prediction[np.argmax(prediction)]

				resp_obj = {
					"verified": verified
					, "score": score
					, "distance": ensemble_features
					, "model": ["PacketNet"]
					, "similarity_metric": ["cosine", "euclidean", "euclidean_l2"]
				}

				if bulkProcess == True:
					resp_objects.append(resp_obj)
				else:
					return resp_obj

			#----------------------

		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)

	#-------------------------

	toc = time.time()

	if bulkProcess == True:

		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["pair_%d" % (i+1)] = resp_item

		return resp_obj


def find(img_path, db_path, model_name ='PacketNet', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base', silent=False):

	"""
	This function applies verification several times and find an identity in a database

	Parameters:
		img_path: exact image path, numpy array (BGR) or based64 encoded image. If you are going to find several identities, then you should pass img_path as array instead of calling find function in a for loop. e.g. img_path = ["img1.jpg", "img2.jpg"]

		db_path (string): You should store some .jpg files in a folder and pass the exact folder path to this.

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

		distance_metric (string): cosine, euclidean, euclidean_l2

		model: built deepface model. A face recognition models are built in every call of find function. You can pass pre-built models to speed the function up.

			model = DeepFace.build_model('VGG-Face')

		enforce_detection (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

		prog_bar (boolean): enable/disable a progress bar

	Returns:
		This function returns pandas data frame. If a list of images is passed to img_path, then it will return list of pandas data frame.
	"""

	tic = time.time()

	img_paths, bulkProcess = functions.initialize_input(img_path)

	#-------------------------------

	if os.path.isdir(db_path) == True:

		if model == None:

			if model_name == 'Ensemble':
				if not silent: print("Ensemble learning enabled")
				models = Boosting.loadModel()

			else: #model is not ensemble
				model = build_model(model_name)
				models = {}
				models[model_name] = model

		else: #model != None
			if not silent: print("Already built model is passed")

			if model_name == 'Ensemble':
				Boosting.validate_model(model)
				models = model.copy()
			else:
				models = {}
				models[model_name] = model

		#---------------------------------------

		if model_name == 'Ensemble':
			model_names = ["PacketNet"]
			metric_names = ['cosine', 'euclidean', 'euclidean_l2']
		elif model_name != 'Ensemble':
			model_names = []; metric_names = []
			model_names.append(model_name)
			metric_names.append(distance_metric)

		#---------------------------------------

		file_name = "representations_%s.pkl" % (model_name)
		file_name = file_name.replace("-", "_").lower()

		if path.exists(db_path+"/"+file_name):

			if not silent: print("WARNING: Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")

			f = open(db_path+'/'+file_name, 'rb')
			representations = pickle.load(f)

			if not silent: print("There are ", len(representations)," representations found in ",file_name)

		else: #create representation.pkl from scratch
			employees = []

			for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
				for file in f:
					if ('.jpg' in file.lower()) or ('.png' in file.lower()):
						exact_path = r + "/" + file
						employees.append(exact_path)

			if len(employees) == 0:
				raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")

			#------------------------
			#find representations for db images

			representations = []

			pbar = tqdm(range(0,len(employees)), desc='Finding representations', disable = prog_bar)

			#for employee in employees:
			for index in pbar:
				employee = employees[index]

				instance = []
				instance.append(employee)

				for j in model_names:
					custom_model = models[j]

					representation = represent(img_path = employee
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)

					instance.append(representation)

				#-------------------------------

				representations.append(instance)

			f = open(db_path+'/'+file_name, "wb")
			pickle.dump(representations, f)
			f.close()

			if not silent: print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")

		#----------------------------
		#now, we got representations for facial database

		if model_name != 'Ensemble':
			df = pd.DataFrame(representations, columns = ["identity", "%s_representation" % (model_name)])
		else: #ensemble learning

			columns = ['identity']
			[columns.append('%s_representation' % i) for i in model_names]

			df = pd.DataFrame(representations, columns = columns)

		df_base = df.copy() #df will be filtered in each img. we will restore it for the next item.

		resp_obj = []

		global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing', disable = prog_bar)
		for j in global_pbar:
			img_path = img_paths[j]

			#find representation for passed image

			for j in model_names:
				custom_model = models[j]

				target_representation = represent(img_path = img_path
					, model_name = model_name, model = custom_model
					, enforce_detection = enforce_detection, detector_backend = detector_backend
					, align = align
					, normalization = normalization
					)

				for k in metric_names:
					distances = []
					for index, instance in df.iterrows():
						source_representation = instance["%s_representation" % (j)]

						if k == 'cosine':
							distance = dst.findCosineDistance(source_representation, target_representation)
						elif k == 'euclidean':
							distance = dst.findEuclideanDistance(source_representation, target_representation)
						elif k == 'euclidean_l2':
							distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))

						distances.append(distance)

					#---------------------------

					if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
						continue
					else:
						df["%s_%s" % (j, k)] = distances

						if model_name != 'Ensemble':
							threshold = dst.findThreshold(j, k)
							df = df.drop(columns = ["%s_representation" % (j)])
							df = df[df["%s_%s" % (j, k)] <= threshold]

							df = df.sort_values(by = ["%s_%s" % (j, k)], ascending=True).reset_index(drop=True)

							resp_obj.append(df)
							df = df_base.copy() #restore df for the next iteration

			#----------------------------------

			if model_name == 'Ensemble':

				feature_names = []
				for j in model_names:
					for k in metric_names:
						if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
							continue
						else:
							feature = '%s_%s' % (j, k)
							feature_names.append(feature)

				#print(df.head())

				x = df[feature_names].values

				#--------------------------------------

				boosted_tree = Boosting.build_gbm()

				y = boosted_tree.predict(x)

				verified_labels = []; scores = []
				for i in y:
					verified = np.argmax(i) == 1
					score = i[np.argmax(i)]

					verified_labels.append(verified)
					scores.append(score)

				df['verified'] = verified_labels
				df['score'] = scores

				df = df[df.verified == True]
				#df = df[df.score > 0.99] #confidence score
				df = df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
				df = df[['identity', 'verified', 'score']]

				resp_obj.append(df)
				df = df_base.copy() #restore df for the next iteration

			#----------------------------------

		toc = time.time()

		if not silent: print("find function lasts ",toc-tic," seconds")

		if len(resp_obj) == 1:
			return resp_obj[0]

		return resp_obj

	else:
		raise ValueError("Passed db_path does not exist!")

	return None

def represent(img_path, model_name = 'PacketNet', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base'):

	"""
	This function represents facial images as vectors.

	Parameters:
		img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.

			model = DeepFace.build_model('VGG-Face')

		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

		normalization (string): normalize the input image before feeding to model

	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
	"""
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# if model_name == "ShuffleFaceNet":
	# 	from deep.basemodels import ShuffleFaceNet
	# 	model = ShuffleFaceNet.loadModel()
	# 	target_size = (112, 112)
	# 	print("ShuffleFaceNet was chosen")

	from deep.basemodels import ShuffleFaceNet
	model = ShuffleFaceNet.loadModel()
	target_size = (112, 112)
	print("ShuffleFaceNet was chosen")

	# elif model_name == "FaceNet":
	# 	from deep.basemodels import FaceNet
	# 	model = FaceNet.loadModel()
	# 	target_size = (112, 112)
	# 	print("FaceNet was chosen")

	# elif model_name == "MobileFaceNet":
	# 	from deep.basemodels import MobileFaceNet
	# 	model = MobileFaceNet.loadModel()
	# 	target_size = (102, 102)
	# 	print("MobileFaceNet was chosen")

	# elif model_name == 'ArcFace':
	# 	from deep.basemodels import ArcFace
	# 	model = ArcFace.loadModel()
	# 	target_size = (112, 112)
	# 	print("ArcFace was chosen")
	
	# elif model_name == 'PocketNet':
	# 	from deep.basemodels import PocketNet
	# 	model = PocketNet.loadModel()
	# 	target_size = (112, 112)
	# 	print("PocketNet was chosen")
	# model_name == 'PocketNet'
	# from deep.basemodels import PocketNet
	# model = PocketNet.loadModel()
	# target_size = (112, 112)
	# print("PocketNet was chosen")
	
	# elif model_name == 'ElasticFace':
	# 	from deep.basemodels import ElasticFace
	# 	model = ElasticFace.loadModel()
	# 	target_size = (112, 112)
	# 	print("ElasticFace was chosen")

	# elif model_name == 'MixFaceNetXS':
	# 	from deep.basemodels import MixFaceNetXS
	# 	model = MixFaceNetXS.loadModel()
	# 	target_size = (112, 112)
	# 	print("MixFaceNetXS was chosen")
	
	# elif model_name == 'MixFaceNetM':
	# 	from deep.basemodels import MixFaceNetM
	# 	model = MixFaceNetM.loadModel()
	# 	target_size = (112, 112)
	# 	print("MixFaceNetM was chosen")


	#detect and align
	img = functions.preprocess_face(img = img_path
		, target_size=target_size
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend
		, align = align)

	# print(img.shape)
	#---------------------------------
	#custom normalization
	
	# Removed from source code
	# img = functions.normalize_input(img = img, normalization = normalization)

	#---------------------------------
	#Convert to tensor

	# img_tensor = torch.Tensor(img)
	# img_tensor = img_tensor.permute(0, 3, 1, 2)
	# img_tensor = img_tensor.to(device)

	# img_tensor = img_tensor.permute(2, 0, 1)
	# if model_name == 'MobileFaceNet':

	# 	img = np.transpose(img, (2, 0, 1))
	# 	img = torch.from_numpy(img).unsqueeze(0).float().to(device)
	# 	img.div_(255).sub_(0.5).div_(0.5)
	# 	# transf = transforms.Compose([Resize(102,102), ToTensor()])
	# 	# img_tensor = transf(img).unsqueeze(0).to(device)

	# elif model_name == 'ArcFace':

	# 	img = np.transpose(img, (2, 0, 1))
	# 	img = torch.from_numpy(img).unsqueeze(0).float().to(device)
	# 	img.div_(255).sub_(0.5).div_(0.5)
	
	# else:
	# 	# img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
	# 	img = np.transpose(img, (2, 0, 1))
	# 	img = torch.from_numpy(img).unsqueeze(0).float().to(device)
	# 	img.div_(255).sub_(0.5).div_(0.5)

	img = np.transpose(img, (2, 0, 1))
	img = torch.from_numpy(img).unsqueeze(0).float()
	img.div_(255).sub_(0.5).div_(0.5)
	img = img.to(device)
	#---------------------------------

	#represent
	embedding = model(img)[0].tolist()
	# embedding = model(img_tensor)[0]
	# embedding = model.predict(img)[0].tolist()

	return embedding

def detectFace(img_path, target_size = (112, 112), detector_backend = 'opencv', enforce_detection = True, align = True):

	"""
	This function applies pre-processing stages of a face recognition pipeline including detection and alignment

	Parameters:
		img_path: exact image path, numpy array (BGR) or base64 encoded image

		detector_backend (string): face detection backends are retinaface, mtcnn, opencv, ssd or dlib

	Returns:
		deteced and aligned face in numpy format
	"""

	img = functions.preprocess_face(img = img_path, target_size = target_size, detector_backend = detector_backend
		, enforce_detection = enforce_detection, align = align)[0] #preprocess_face returns (1, 112, 112, 3)
	return img[:, :, ::-1] #bgr to rgb

#---------------------------
#main

functions.initialize_folder()

def cli():
	import fire
	fire.Fire()
