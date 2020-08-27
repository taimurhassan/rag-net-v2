import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset, ImageSegmentationGen
import os
import glob
import six
import matplotlib.pyplot as plt
from keras import optimizers
from keras import backend as K
import keras 
import pdb
import tensorflow as tf
import tensorflow.keras.backend as keras_backend

def loss_function(y, pred_y):
    #return keras_backend.mean(keras.losses.categorical_crossentropy(y, pred_y))
    return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model(tf.Variable(x))
    mse = loss_fn(y, logits)
    return mse, logits
    
def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
def customLoss(yTrue,yPred):
    alpha1 = 0.5
    alpha2 = 0.5
    dice = K.mean(1-(2 * K.sum(yTrue * yPred))/(K.sum(yTrue + yPred)))
    return alpha1 * dice + alpha2 * keras.losses.categorical_crossentropy(yTrue,yPred)
	
    
def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          optimizer_name='adadelta' , 
		  do_augment=False, 
		  classifier=None
          ):

    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None
#loss="categorical_crossentropy",#
    if optimizer_name is not None:
        model.compile(loss=lambda yTrue, yPred: customLoss(yTrue, yPred),
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
            assert verified

    train_gen = ImageSegmentationGen(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width , do_augment=do_augment )
    #pdb.set_trace()
    
    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            
            #obj = iter(train_gen) 
            #try: 
            #while True: # Print till error raised 
                #inputa, labela = next(iter(train_gen))
                #train_maml(model, 1, inputa, labela)
            #except:  
                # when StopIteration raised, Print custom message 
                #print ("\nTraining Completed")  
            
            history = model.fit_generator(train_gen, steps_per_epoch, epochs=1, workers=1)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))		  
            
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=200,  epochs=1, workers=1)
           
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep) 

def copy_model(model):
    model2 = resnet50_segnet(n_classes=model.n_classes,  input_height=model.input_height, input_width=model.input_width)
    model2.set_weights(model.get_weights())
    return model2
    
    
def train_maml(model, epochs, inputa, labela, lr_inner=0.01, batch_size=1, log_steps=1000):
    '''Train using the MAML setup.

    The comments in this function that start with:

        Step X:

    Refer to a step described in the Algorithm 1 of the paper.

    Args:
        model: A model.
        epochs: Number of epochs used for training.
        dataset: A dataset used for training.
        lr_inner: Inner learning rate (alpha in Algorithm 1). Default value is 0.01.
        batch_size: Batch size. Default value is 1. The paper does not specify
            which value they use.
        log_steps: At every `log_steps` a log message is printed.

    Returns:
        A strong, fully-developed and trained maml.
    '''
    optimizer = keras.optimizers.Adam()

    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        #start = time.time()
        # Step 3 and 4
        x, y = inputa, labela
        #a = tf.convert_to_tensor(x)
        #print(a.numpy())
        model(tf.Variable(x))  # run forward pass to initialize weights

        with tf.GradientTape() as test_tape:
            test_tape.watch(model.trainable_weights)
            # Step 5
            with tf.GradientTape() as train_tape:
                train_loss, _ = compute_loss(model, x, y)
            # Step 6
            
            print(train_loss)
            pdb.set_trace()
            gradients = train_tape.gradient(train_loss, tf.Variable(model.trainable_weights))
            
            model_copy = copy_model(model)

            for j in range(len(model_copy.trainable_weights)):
                model_copy.trainable_weights[j] = tf.subtract(model.trainable_weights[j],
                                                                    tf.multiply(lr_inner, gradients[j]))
            # Step 8
            test_loss, logits = compute_loss(model_copy, x, y)
        
            # Step 10
        gradients = test_tape.gradient(test_loss, model_copy.trainable_weights)
        #pdb.set_trace()
        print(test_loss)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        total_loss += test_loss
                    