#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
import tensorflow as tf
import argparse
import configparser

import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt

import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'customModel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'datasetPreparation'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from customModel import customModel
from datasetPreparation import datasetPreparation

#warnings.filterwarnings("ignore")


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
    parser.add_argument('--net', default='', help="path to net (to continue training)")
    parser.add_argument('--outf', default='tmp', help='folder to output images and model checkpoints, it will add a train_ in front of the name')
    parser.add_argument('--namefile', default='epoch', help="name to put on the file of the save weights")
    parser.add_argument('--sigma', default=4, help='keypoint creation size for sigma')
    parser.add_argument('--savedmodelpath', help='path to the saved model')

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
    debugFolderPath = os.path.join(dir_path, 'test_DEBUG', str(opt.outf))

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
    # save the epoch loss, used later to print the loss in a matplotlib
    beliefLoss_perBatch = []
    affinityLoss_perBatch = []
    
    with tf.device('/GPU:0'):

        try:
            tf.print ("START: {}".format(datetime.datetime.now().time()))
            tf.print('using tensorflow version: ' + str(tf.__version__))

            tf.keras.backend.clear_session()

            # netModel = customModel(pretrained=True, blocks=6, freezeLayers=14,)
            tf.print('Loading the model from path:\n{}'.format(opt.savedmodelpath))
            netModel = tf.keras.models.load_model(opt.savedmodelpath)

            # model can be built by calling the build function but then all of the layers have to be used.
            # or by calling the fit function
            # to load weights model has to be built
            # tf.print('building model: {}'.format(netModel.name))
            # netModel.build(input_shape=(None, 400, 400, 3))

            #tf.print('loading weights from: {}'.format(opt.ckptpath))
            #netModel.load_weights(filepath=opt.ckptpath)
            # netModel = tf.keras.models.load_model(filepath=opt.net)

            # Instantiate losses and metrics.
            loss_fn_beliefs = tf.keras.losses.MeanSquaredError(name='beliefs_loss')
            loss_fn_affinities = tf.keras.losses.MeanSquaredError(name='affinities_loss')

            beliefs_metric = tf.keras.metrics.MeanSquaredError(name='beliefs_accuracy', dtype=tf.float32) 
            affinities_metric = tf.keras.metrics.MeanSquaredError(name='affinities_accuracy', dtype=tf.float32)

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

            # Iterate over the batches of the dataset.
            for step, batch_test in enumerate(test_dataset):
                # Run the forward pass of the layer.
                logits_beliefs, logits_affinities = netModel(batch_test[0]['images'], training=False,)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_beliefs_value = loss_fn_beliefs(y_true=batch_test[1]['beliefs'], y_pred=logits_beliefs)
                loss_affinities_value = loss_fn_affinities(y_true=batch_test[1]['affinities'], y_pred=logits_affinities)
                loss = loss_beliefs_value + loss_affinities_value

                #append loss values in list for later
                beliefLoss_perBatch.append(loss_beliefs_value)
                affinityLoss_perBatch.append(loss_affinities_value)

                # metrics
                beliefs_metric.update_state(y_true=batch_test[1]['beliefs'], y_pred=logits_beliefs)
                affinities_metric.update_state(y_true=batch_test[1]['affinities'], y_pred=logits_affinities)

                # update the progress bar in the epoch
                # jupyter progress bar creates a new line after each update
                pbar.update(step+1, values=[('mse_beliefs', beliefs_metric.result()), ('mse_affinities', affinities_metric.result()),])
                # tf.print('')
                
                #################################################################################################################
                ################################################# print outputs #################################################
                # show labels and logits of the first image of the dataset
                imgBelLog, imgBelLab, imgAffLog, imgAffLab = getLabelsLogitsImages(logits_beliefs, logits_affinities, batch_test, 0)

                list_im = [imgBelLog, imgBelLab, imgAffLog, imgAffLab]
                width = imgBelLog.width
                height = imgBelLog.height
                new_im_width = len(list_im)*(width)
                # creates a new empty image, RGB mode, and size 444 by 95
                new_img = Image.new('RGB', (new_im_width, height+10))
                
                for place, elem in enumerate(list_im):
                    im=elem
                    new_img.paste(im, (place*width,0))

                new_img.save(os.path.join(debugFolderPath, 'test_{}.jpg'.format(step)))
                #new_img.show()

            #Beliefs loss
            tf.print("SAVING BELIEFS AND AFFINITIES LOSS PLOTS...")
            plt.figure(figsize=(20,10))
            plt.subplot(1, 2, 1)
            plt.title('Beliefs loss')
            plt.xlabel("img nr.")
            plt.ylabel("loss")
            plt.plot(beliefLoss_perBatch)
            
            #Affinities loss
            plt.subplot(1, 2, 2)
            plt.title('Affinities loss')
            plt.xlabel("img nr.")
            plt.ylabel("loss")
            plt.plot(affinityLoss_perBatch)
            plt.savefig(os.path.join(debugFolderPath, 'test_affinities&beliefs_loss.png'), bbox_inches='tight')

        except ImportError as ie:
            tf.print('import error:\n {}'.format(ie))

        except ValueError as ve:
            tf.print('value error:\n {}'.format(ve))

        except:
            e = sys.exc_info()[0]
            tf.print('exception: {}'.format(e))