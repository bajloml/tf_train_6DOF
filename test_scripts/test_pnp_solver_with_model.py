#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
import tensorflow as tf
import argparse
import configparser
import cv2

import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import math

import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'customModel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'datasetPreparation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'positionSolver'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from featureModel import featureModel
from residualModel import residualModel
from positionSolver import positionSolver

from datasetPreparation import datasetPreparation

#warnings.filterwarnings("ignore")

def saveLabelLogits(img1, img2, img3, img4, filename):

    fig = plt.figure(figsize=(10, 5))
    rows = 1
    columns = 4

    fig.add_subplot(rows, columns, (1))
    plt.axis('off')
    plt.imshow(img1)
    plt.title("beliefs logits")

    fig.add_subplot(rows, columns, (2))
    plt.axis('off')
    plt.imshow(img2)
    plt.title("beliefs label")

    fig.add_subplot(rows, columns, (3))
    plt.axis('off')
    plt.imshow(img3)
    plt.title("affinity logits")

    fig.add_subplot(rows, columns, (4))
    plt.axis('off')
    plt.imshow(img4)
    plt.title("affinity label")
    plt.savefig(filename+'.png')

def getLabelsLogitsImages(logits_bel, logits_aff, batch_tr, img_number):

    # belief images points
    beliefLogitsTensor = tf.reduce_sum(logits_bel[img_number], axis=2) * 255
    beliefLogitsNumpy = beliefLogitsTensor.numpy()
    img1 = Image.fromarray(beliefLogitsNumpy)

    beliefLabelTensor =  tf.reduce_sum(batch_tr[1]['beliefs'][img_number], axis=2) * 255
    beliefLabelNumpy = beliefLabelTensor.numpy()
    img2 = Image.fromarray(beliefLabelNumpy)

    affinityLogitsTensor = tf.reduce_sum(logits_aff[img_number], axis=2)
    affinityLogitsNumpy = affinityLogitsTensor.numpy()
    img3 = Image.fromarray(affinityLogitsNumpy)

    affinityLabelTensor =  tf.reduce_sum(batch_tr[1]['affinities'][img_number], axis=2)
    affinityLabelNumpy = affinityLabelTensor.numpy()
    img4 = Image.fromarray(affinityLabelNumpy)

    return img1, img2, img3, img4

def getFilterImages(logits, img_number):

    filters = tf.shape(logits)[-1] #len(logits[0,50,50])
    images = []
    for filter in range(filters):
        LogitsTensor = logits[img_number,:,:,filter]# * 255
        LogitsNumpy = LogitsTensor.numpy()
        images.append(Image.fromarray(LogitsNumpy))  

    return images

def saveFilterImages(images, filename, polarize=False):

    columns = math.sqrt(len(images))
    rows = columns

    fig = plt.figure(figsize=(rows, columns))
    borderColor = [255, 255, 255]

    list_pol_img = []
    list_img = []

    for row in range(int(rows)):
        for col in range(int(columns)):
            img_index = row*int(rows)+col+1
            fig.add_subplot(rows, columns, img_index)
            plt.axis('off')

            #get numpy array from Image PIL
            img_np = np.array(images[img_index-1])
            list_img.append(img_np)
            #initialy polarized image is same as image
            img_np_pol = img_np

            #polarize everything with treshold of img_np.max value
            tresh_val = int(np.amax(img_np))
            #define lambda function
            polarizator = lambda x: 0 if (x<1*tresh_val) else 255

            if polarize and tresh_val>1:
                #define functor
                vfunc = np.vectorize(polarizator)
                #apply functor to get a polarized image
                img_np_pol = vfunc(img_np)
            #append polarized images to list
            list_pol_img.append(img_np_pol)
            
            #add border to image
            borderedImage = cv2.copyMakeBorder(img_np_pol, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=borderColor)
            plt.imshow(borderedImage)

            #Image.fromarray(img_np_pol).show()
            #Image.fromarray(img_np).show()

    image_pol_array = np.expand_dims(list_pol_img, axis=3)
    pol = np.concatenate(image_pol_array, axis=2)
    pol = np.sum(pol, axis=2)

    image_array = np.expand_dims(list_img, axis=3)
    img = np.concatenate(image_array, axis=2)
    img = np.sum(img, axis=2)
    
    Image.fromarray(img).show()
    Image.fromarray(pol).show()

    plt.savefig(filename+'.png')

if __name__ == '__main__':

    # description string, printed with --help
    config_ini_description = ''.join(("create the .ini file, fill the file:>",
                                      "\n",
                                      "\n[defaults]",
                                      "\n\t datatest = path to the testing data",
                                      "\n\t object = name of the object of interest in the dataset",
                                      "\n\t workers = number of data loading workers",
                                      "\n\t batchsize = input batch size",
                                      "\n\t imagesize = the height / width of the input image to network",
                                      "\n\t noise = gaussian noise added to the image(contrast and brightness)",
                                      "\n\t net = path to net (to continue training on previuosly trained network)",
                                      "\n\t namefile = name to put on the file of the save weights(trained network)",
                                      "\n\t sigma = keypoint creation size for sigma for belif maps",
                                      "\n",
                                      "\n\t and call the script with the arg --config <path to the ini file>"))
    
    # read the parameters given from the powershell/terminal if there are any, define the type of each arg:
    parser = argparse.ArgumentParser(description=config_ini_description, add_help=True) # printed with -h/--help
    parser.add_argument('--datatest', default="", help='path to data testing set')
    parser.add_argument('--object', default=None, help='In the dataset which objet of interest')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--imagesize', type=int, default=400, help='the height width of the input image to network')
    parser.add_argument('--noise', type=float, default=2.0, help='gaussian noise added to the image')
    parser.add_argument('--model', default="", help='model which will be used for predicition')
    parser.add_argument('--ckptPath', default='', help="path to model checkpoints")
    parser.add_argument('--outf', default='tmp', help='folder to output images and model checkpoints, it will add a train_ in front of the name')
    parser.add_argument('--namefile', default='epoch', help="name to put on the file of the save weights")
    parser.add_argument('--sigma', default=4, help='keypoint creation size for sigma')

    # Read the config but do not overwrite the args written 
    # read the configuration from the file given by with a "-c" or "--config" file if it exists
    conf_parser = argparse.ArgumentParser( description=__doc__, # printed with -h/--help
                                           # Don't mess with format of description
                                           formatter_class=argparse.RawDescriptionHelpFormatter,
                                           # Turn off help, so we print all options in response to -h
                                           add_help=True)

    conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {"option":"default"}

    if args.config:
        config = configparser.SafeConfigParser()
        config.read([args.config])
        # add defaults from the .ini file to the defaults dictionary
        defaults.update(dict(config.items("defaults")))

    # set defaults from the dictionary defaults
    parser.set_defaults(**defaults)
    parser.add_argument("--option")
    opt = parser.parse_args(remaining_argv)

    # transform dictionary to apply on the image in the train dataset preparation
    contrast = 0.2
    brightness = 0.2
    transform = {'contrast': contrast,
                 'brightness': brightness,
                 'imgSize': int(opt.imagesize)}
    #########################################Create debug folder to store the debug data##################################################
    dir_path = os.path.dirname(os.path.realpath(__file__))
    debugFolderPath = os.path.join(dir_path, 'DEBUG_pnp_solver_{}'.format(opt.model), str(opt.outf))

    # if debug folder doesn't exists, create it
    if not os.path.isdir(debugFolderPath):
        try:
            os.makedirs(debugFolderPath)
        except OSError:
            pass

    if opt.datatest != "":
        test_dataset = datasetPreparation(root=opt.datatest, objectsofinterest=opt.obj, datasetName='test_Dataset',
                                          batch_size=opt.batchsize, keep_orientation=True, noise=opt.noise,
                                          sigma=opt.sigma, debugFolderPath=debugFolderPath,
                                          transform=transform, shuffle=True, saveAffAndBelImages=False)
    
    with tf.device('/GPU:0'):

        try:
            tf.print ("START: {}".format(datetime.datetime.now().time()))
            tf.print('using tensorflow version: ' + str(tf.__version__))

            tf.keras.backend.clear_session()

            if(opt.model=='featureModel'):
                tf.print('Creating feature model')
                netModel = featureModel(pretrained=True, blocks=6, numFeatures=512, freezeLayers=14,)
            elif(opt.model=='residualModel'):
                tf.print('Creating residual model')
                netModel = residualModel(pretrained=True, blocks=6, freezeLayers=14,)

            # model can be built by calling the build function but then all of the layers have to be used.
            # or by calling the fit function
            # to load weights model has to be built
            tf.print('building model: {}'.format(netModel.name))
            netModel.build(input_shape=(None, 400, 400, 3))

            tf.print('loading weights from: {}'.format(opt.ckptpath))
            netModel.load_weights(filepath=opt.ckptpath)
            # netModel = tf.keras.models.load_model(filepath=opt.net)

            # check dataset
            batch_check = test_dataset[0]
            tf.print('batch_check[0][images] shape-> {}'.format(tf.shape(batch_check[0]['images'])))
            tf.print('batch_check[1][beliefs] shape-> {}'.format(tf.shape(batch_check[1]['beliefs'])))
            tf.print('batch_check[1][affinities] shape-> {}'.format(tf.shape(batch_check[1]['affinities'])))
            tf.print('')
            tf.print('batch_check[0][images] max-> {}, min-> {}'.format(tf.reduce_max(batch_check[0]['images']), tf.reduce_min(batch_check[0]['images'])))
            tf.print('batch_check[1][beliefs] max-> {}, min-> {}'.format(tf.reduce_max(batch_check[1]['beliefs']), tf.reduce_min(batch_check[1]['beliefs'])))
            tf.print('batch_check[1][affinities] max-> {}, min-> {}'.format(tf.reduce_max(batch_check[1]['affinities']), tf.reduce_min(batch_check[1]['affinities'])))

            # progress bar, used to show the progress of each epoch
            pbar = tf.keras.utils.Progbar(len(test_dataset), width=80, stateful_metrics=['mse_beliefs', 'mse_affinities',])

            #position solver
            if opt.debug == 'True':
                debug = True
            else:
                debug = False

            ps_true = positionSolver(opt.camsettings, opt.objsettings, debug, text_width_ratio=0.01, text_height_ratio=0.05, 
                                    text = 'Label',  belColor = (255, 0, 0), affColor = (255, 0, 0)) 
            ps_prediction = positionSolver(opt.camsettings, opt.objsettings, debug, text_width_ratio=0.01, text_height_ratio=0.1, 
                                    text = 'Logit',  belColor = (0, 255, 0), affColor = (0, 255, 0)) 

            # Iterate over the batches of the dataset.
            for step, batch_test in enumerate(test_dataset):
                # Run the forward pass of the layer.
                logits_beliefs, logits_affinities = netModel(batch_test[0]['images'], training=False,)  # Logits for this minibatch
                test_img = batch_test[0]['images']
                # true values
                rot_true, tran_true, test_img, _ = ps_true.getPosition(batch_test[1]['beliefs'], batch_test[1]['affinities'], test_img)
                #convert numpy array to tensor
                test_img = tf.convert_to_tensor(test_img, dtype=tf.float32)
                #model prediciton
                rot_pred, tran_pred, test_img, _ = ps_prediction.getPosition(logits_beliefs, logits_affinities, test_img)
                
                #true rotation and translation
                tf.print("Label ROTATION and TRANSLATION: ")
                tf.print("ROTATION:")
                {tf.print('\t {}'.format(value)) for value in rot_true}
                tf.print("TRANSLATION:")
                {tf.print('\t {}'.format(value)) for value in tran_true}
                #predicted rotation and translation
                tf.print("Logit ROTATION and TRANSLATION: ")
                tf.print("ROTATION:")
                {tf.print('\t {}'.format(value)) for value in rot_pred}
                tf.print("TRANSLATION:")
                {tf.print('\t {}'.format(value)) for value in tran_pred}
                #save image
                test_img = Image.fromarray(test_img)
                test_img.save(os.path.join( debugFolderPath, 'test_{}.png'.format(step)))

        except ImportError as ie:
            tf.print('import error:\n {}'.format(ie))

        except ValueError as ve:
            tf.print('value error:\n {}'.format(ve))
        
        except cv2.error as e:
            tf.print('opencv error:\n {}'.format(e))

        except:
            e = sys.exc_info()[0]
            tf.print('exception: {}'.format(e))