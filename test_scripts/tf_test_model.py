#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
import tensorflow as tf

import numpy as np
import datetime
from PIL import Image

import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'customModel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'datasetPreparation'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from customModel import customModel
from datasetPreparation import datasetPreparation

warnings.filterwarnings("ignore")


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


if __name__ == '__main__':

    datatest      = 'path to the dataset of the image to test on (used to compare labels and logits)'
    testImage     = 'path to image to test on'
    obj           = 'thorHammer_387'
    batchsize     = 15
    imagesize     = 400
    noise         = 2.0
    ckptPath      = 'path to saved model checkpoints'
    namefile      = 'ModelFromGCP'
    sigma         = 5

    debugFolderPath = 'path to folder where the debug info should be saved'

    # transform dictionary to apply on the image in the train dataset preparation
    contrast = 0.2
    brightness = 0.2
    transform = {'contrast': contrast,
                 'brightness': brightness,
                 'imgSize': int(imagesize)}

    if datatest != "":
        test_dataset = datasetPreparation(root=datatest, objectsofinterest=obj, datasetName='test_Dataset',
                                          batch_size=batchsize, keep_orientation=True, noise=noise,
                                          sigma=sigma, debugFolderPath=debugFolderPath,
                                          transform=transform, shuffle=True, saveAffAndBelImages=False)
    

    with tf.device('/GPU:0'):

        try:
            tf.print ("START: {}".format(datetime.datetime.now().time()))
            tf.print('using tensorflow version: ' + str(tf.__version__))

            tf.keras.backend.clear_session()

            netModel = customModel(pretrained=True, blocks=6, freezeLayers=14,)

            # model can be built by calling the build function but then all of the layers have to be used.
            # or by calling the fit function
            # to load weights model has to be built
            tf.print('building model: {}'.format(netModel.name))
            netModel.build(input_shape=(None, 400, 400, 3))

            tf.print('loading weights from: {}'.format(ckptPath))
            netModel.load_weights(filepath=ckptPath)
            # netModel = tf.keras.models.load_model(filepath=opt.net)

            # read the image from the drive:
            img = Image.open(testImage)
            img.show()
            img_np = np.array(img)
            # do not include forth channel is transparency from .png
            img_tensor = tf.convert_to_tensor(img_np[:,:,:3], dtype=tf.float32)
            # expand the dim at pos 0(batch size)
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            # Run the forward pass and calculate the time for a pass.
            startTime = datetime.datetime.now().time()
            logits_beliefs, logits_affinities = netModel(img_tensor)  # Logits for this minibatch
            endTime = datetime.datetime.now().time()
            duration = datetime.datetime.combine(datetime.date.today(), endTime) - datetime.datetime.combine(datetime.date.today(), startTime)
            tf.print('forward pass duration: {}'.format(duration))

            #################################################################################################################
            ################################################# print outputs #################################################

            # show labels and logits of the first image of the dataset
            imgBelLog, imgBelLab, imgAffLog, imgAffLab = getLabelsLogitsImages(logits_beliefs, logits_affinities, test_dataset[0], 0)
            imgBelLog.show()
            imgBelLab.show()
            imgAffLog.show()
            imgAffLab.show()

            tf.print('END: {}'.format(datetime.datetime.now().time())) 


        except ImportError as ie:
            tf.print('import error:\n {}'.format(ie))

        except ValueError as ve:
            tf.print('value error:\n {}'.format(ve))

        except:
            e = sys.exc_info()[0]
            tf.print('exception: {}'.format(e))


