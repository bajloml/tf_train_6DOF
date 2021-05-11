#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
# import pydot
import argparse
import configparser
import datetime
import warnings
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    tf.print('MEMORY GROWTH ENABLED')
else:
    tf.print('CPU USED -- NO MEMORY GROWTH ENABLED')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'customModel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'datasetPreparation'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

# from customModel import customModel
from featureModel import featureModel
from datasetPreparation import datasetPreparation

tf.get_logger().setLevel('WARNING')
tf.print('using tensorflow version: {}'.format(str(tf.__version__)))
tf.print('eager execution enabled: {}'.format(tf.executing_eagerly()))

warnings.filterwarnings("ignore")


"""
DOPE network
@inproceedings{tremblay2018corl:dope,
    author = {Jonathan Tremblay and Thang To and Balakumar Sundaralingam and Yu Xiang and Dieter Fox and Stan Birchfield},
    title = {Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects},
    booktitle = {Conference on Robot Learning (CoRL)},
    url = "https://arxiv.org/abs/1809.10790",
    year = 2018
    }

Dataset_Synthesizer
@misc{to2018ndds,
    author = {Thang To and Jonathan Tremblay and Duncan McKay and Yukie Yamaguchi and Kirby Leung
            and Adrian Balanon and Jia Cheng and William Hodge and Stan Birchfield},
    url = "https://github.com/NVIDIA/Dataset_Synthesizer",
    title = {{NDDS}: {NVIDIA} Deep Learning Dataset Synthesizer},
    Year = 2018
    }

train with Tensorflow
virtual env with the python3.8 is used the package
to use this script on the PC with multiple versions of the python, either use the venv

"""


def getLabelsLogitsImages(logits_bel, logits_aff, batch, img_number):

    # belief logits
    beliefLogitsTensor = tf.reduce_sum(logits_bel[img_number], axis=2)  # * 255
    beliefLogitsNumpy = beliefLogitsTensor.numpy().astype(np.uint8)
    belLog_img = Image.fromarray(beliefLogitsNumpy)

    # belief label
    beliefLabelTensor = tf.reduce_sum(batch[1]['beliefs'][img_number], axis=2)  # * 255
    beliefLabelNumpy = beliefLabelTensor.numpy().astype(np.uint8)
    belLab_img = Image.fromarray(beliefLabelNumpy)

    # affinity logits
    affinityLogitsTensor = tf.reduce_sum(logits_aff[img_number], axis=2)  # * 255
    affinityLogitsNumpy = affinityLogitsTensor.numpy().astype(np.uint8)
    affLog_img = Image.fromarray(affinityLogitsNumpy)

    # affinity label
    affinityLabelTensor = tf.reduce_sum(batch[1]['affinities'][img_number], axis=2)  # * 255
    affinityLabelNumpy = affinityLabelTensor.numpy().astype(np.uint8)
    affLab_img = Image.fromarray(affinityLabelNumpy)

    return belLog_img, belLab_img, affLog_img, affLab_img


def saveLabelLogits(belLog_img, belLab_img, affLog_img, affLab_img, filename):

    fig = plt.figure(figsize=(10, 5))
    rows = 1
    columns = 4

    fig.add_subplot(rows, columns, (1))
    plt.axis('off')
    plt.imshow(belLog_img)
    plt.title("beliefs logits")

    fig.add_subplot(rows, columns, (2))
    plt.axis('off')
    plt.imshow(belLab_img)
    plt.title("beliefs label")

    fig.add_subplot(rows, columns, (3))
    plt.axis('off')
    plt.imshow(affLog_img)
    plt.title("affinity logits")

    fig.add_subplot(rows, columns, (4))
    plt.axis('off')
    plt.imshow(affLab_img)
    plt.title("affinity label")
    plt.savefig(filename+'.png')


def forwardPass(model, dataset, optimizer, loss_fn_belief, loss_fn_affinity, belief_metric, affinity_metric, epoch, pathToSave, training):
    """
    run the forward pass on the dataset, returns the model, losses and metric
    training input differs pass on train from the pass on test data
    """

    # save the epoch loss, used later to print the loss in a matplotlib
    beliefLoss_list = []
    affinityLoss_list = []

    # Iterate over the batches of the dataset.
    for step, batch in enumerate(dataset):

        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:

            # Run the forward pass of the layer to get logits for this minibatch
            logits_beliefs, logits_affinities = model(batch[0]['images'], training=training)

            # logits_beliefs_numpy = logits_beliefs.numpy()
            # logits_affinities_numpy = logits_affinities.numpy()
            # labels_beliefs_numpy = batch_train[1]['beliefs'].numpy()
            # labels_affinities_numpy = batch_train[1]['affinities'].numpy()

            # Compute the loss value for this minibatch.
            loss_beliefs_value = loss_fn_belief(y_true=batch[1]['beliefs'], y_pred=logits_beliefs)
            loss_affinities_value = loss_fn_affinity(y_true=batch[1]['affinities'], y_pred=logits_affinities)
            loss = loss_beliefs_value + loss_affinities_value

        # metrics
        belief_metric.update_state(y_true=batch[1]['beliefs'], y_pred=logits_beliefs)
        affinity_metric.update_state(y_true=batch[1]['affinities'], y_pred=logits_affinities)

        if training:
            # pbar is not used, because of the nohup
            tf.print('train: {}/{}, mse beliefs: {}, mse_affinity: {}'.format(step+1, len(dataset), loss_beliefs_value, loss_affinities_value))
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, model.trainable_variables)
            # apply gradients through backpropagation
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            # pbar is not used, because of the nohup
            tf.print('test: {}/{}, mse beliefs: {}, mse_affinity: {}'.format(step+1, len(dataset), loss_beliefs_value, loss_affinities_value))

        beliefLoss_list.append(loss_beliefs_value)
        affinityLoss_list.append(loss_affinities_value)

    del(tape)

    # at the end save some labels and logits
    # show labels and logits of the first image of the batch
    belLog_img, belLab_img, affLog_img, affLab_img = getLabelsLogitsImages(logits_beliefs, logits_affinities, batch, 0)

    if training:
        # save images
        saveLabelLogits(belLog_img, belLab_img, affLog_img, affLab_img, os.path.join(pathToSave, 'train_{}_epoch.png'.format(epoch)))
    else:
        # save images
        saveLabelLogits(belLog_img, belLab_img, affLog_img, affLab_img, os.path.join(pathToSave, 'test_{}_epoch.png'.format(epoch)))

    return model, belief_metric, affinity_metric, beliefLoss_list, affinityLoss_list


##################################################
# TRAINING CODE MAIN STARTING HERE
##################################################

if __name__ == '__main__':

    tf.print("start time:{}".format(datetime.datetime.now().time()))

    tf.print('arguments passed: {}'.format(str(sys.argv)))

    # description string, printed with --help
    config_ini_description = ''.join(("create the .ini file, fill the file:>",
                                      "\n",
                                      "\n[defaults]",
                                      "\n\t data = path to the training data",
                                      "\n\t datatest = path to the testing data",
                                      "\n\t object = name of the object of interest in the dataset",
                                      "\n\t batchsize = input batch size",
                                      "\n\t imagesize = the height / width of the input image to network",
                                      "\n\t lr = learning rate, default=0.001",
                                      "\n\t noise = gaussian noise added to the image(contrast and brightness)",
                                      "\n\t net = path to net (to continue training on previuosly trained network)",
                                      "\n\t namefile = name to put on the file of the save weights(trained network)",
                                      "\n\t epochs = number of epochs to train"
                                      "\n\t gpuids = NVIDIA GPUs to use with CUDA, with AMD we use plaidml and keras",
                                      "\n\t outf = folder to output images and model checkpoints, it will add a train_ in front of the name",
                                      "\n\t sigma = keypoint creation size for sigma for belif maps",
                                      "\n\t pretrained = do you want to use vgg imagenet pretrained weights",
                                      "\n\t savemodelafterepoch = should model  be saved after each epoch",
                                      "\n\t usetestdataset = should the model be tested after every train epoch",
                                      "\n",
                                      "\n\t and call the script with the arg --config <path to the ini file>"))

    # read the parameters given from the powershell/terminal if there are any, define the type of each arg:
    parser = argparse.ArgumentParser(description=config_ini_description, add_help=True)  # printed with -h/--help
    parser.add_argument('--data', default="", help='path to training data')
    parser.add_argument('--datatest', default="", help='path to data testing set')
    parser.add_argument('--object', default=None, help='In the dataset which objet of interest')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--imagesize', type=int, default=400, help='the height width of the input image to network')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
    parser.add_argument('--noise', type=float, default=2.0, help='gaussian noise added to the image')
    parser.add_argument('--net', default='', help="path to net (to continue training)")
    parser.add_argument('--namefile', default='epoch', help="name to put on the file of the save weights")
    parser.add_argument('--epochs', type=int, default=60, help="number of epochs to train")
    parser.add_argument('--gpuids', nargs='+', type=int, default=[0], help='GPUs to use')
    parser.add_argument('--outf', default='tmp', help='folder to output images and model checkpoints, it will add a train_ in front of the name')
    parser.add_argument('--sigma', default=5, help='keypoint creation size for sigma')
    parser.add_argument("--pretrained", type=bool, default=True, help='do you want to use vgg imagenet pretrained weights')
    parser.add_argument("--savemodelafterepoch", type=bool, default=True, help='should model and the checkpoints be saved after each epoch')
    parser.add_argument("--usetestdataset", type=bool, default=True, help='should the model be tested after every train epoch')

    # Read the config but do not overwrite the args written
    # read the configuration from the file given by with a "-c" or "--config" file if it exists
    conf_parser = argparse.ArgumentParser(description=__doc__,  # printed with -h/--help
                                          # Don't mess with format of description
                                          formatter_class=argparse.RawDescriptionHelpFormatter,
                                          # Turn off help, so we print all options in response to -h
                                          add_help=True)

    conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {"option": "default"}

    if args.config:
        config = configparser.SafeConfigParser()
        config.read([args.config])
        # add defaults from the .ini file to the defaults dictionary
        defaults.update(dict(config.items("defaults")))

    # set defaults from the dictionary defaults
    parser.set_defaults(**defaults)
    parser.add_argument("--option")
    opt = parser.parse_args(remaining_argv)

    # ########################################Create debug folder to store the debug data##################################################
    dir_path = os.path.dirname(os.path.realpath(__file__))
    debugFolderPath = os.path.join(dir_path, 'DEBUG_{}'.format(str(opt.outf)))

    # if debug folder doesn't exists, create it
    if not os.path.isdir(debugFolderPath):
        try:
            os.makedirs(debugFolderPath)
        except OSError:
            pass

    # dataset images path
    trainDatasetImagePath = os.path.join(debugFolderPath, 'dataset_imgs', 'train')
    if not os.path.isdir(trainDatasetImagePath):
        try:
            os.makedirs(trainDatasetImagePath)
        except OSError:
            pass

    testDatasetImagePath = os.path.join(debugFolderPath, 'dataset_imgs', 'test')
    if not os.path.isdir(testDatasetImagePath):
        try:
            os.makedirs(testDatasetImagePath)
        except OSError:
            pass

    # make dir to save logits_labels_img_epoch_end images
    labelsLogitsImgPath = os.path.join(debugFolderPath, 'dataset_imgs', 'logits_labels_img_epoch_end')
    if not os.path.isdir(labelsLogitsImgPath):
        try:
            os.makedirs(labelsLogitsImgPath)
        except OSError:
            pass

    # make folder for ckpts and models
    modelFolder = os.path.join(debugFolderPath, 'models')
    ckptFolder = os.path.join(debugFolderPath, 'ckpt')

    if not os.path.isdir(modelFolder):
        try:
            os.makedirs(modelFolder)
        except OSError:
            pass

    if not os.path.isdir(ckptFolder):
        try:
            os.makedirs(ckptFolder)
        except OSError:
            pass
    # ##################################################################################################

    # save passed parameters
    if (os.path.exists(os.path.join(debugFolderPath, 'header.txt'))):
        with open(os.path.join(debugFolderPath, 'header.txt'), 'a') as file:
            file.write(str(opt)+"\n")
    else:
        with open(os.path.join(debugFolderPath, 'header.txt'), 'w') as file:
            file.write(str(opt)+"\n")

    if (os.path.exists(os.path.join(debugFolderPath, 'test_metric.csv'))):
        with open(os.path.join(debugFolderPath, 'test_metric.csv'), 'a') as file:
            file.write("epoch, passed,total \n")
    else:
        with open(os.path.join(debugFolderPath, 'test_metric.csv'), 'w') as file:
            file.write("epoch, passed,total \n")
    # print parameters
    print("PARAMETERS:\n")
    parametersList = str(opt).split(',')
    for param in parametersList:
        print(param)

    # transform dictionary to apply on the image in the train dataset preparation
    contrast = 0.2
    brightness = 0.2
    transform = {"contrast": contrast,
                 "brightness": brightness,
                 "imgSize": int(opt.imagesize)}

    # load the datasets using the loader in utils_pose
    train_dataset = test_dataset = None
    tf.print('')
    tf.print('CREATING THE DATASETS:')

    if opt.data != "":
        train_dataset = datasetPreparation(root=opt.data, objectsofinterest=opt.object, datasetName='train_Dataset',
                                           batch_size=opt.batchsize, keep_orientation=True, noise=opt.noise, sigma=opt.sigma,
                                           debugFolderPath=trainDatasetImagePath, transform=transform, shuffle=True, saveAffAndBelImages=False)

    if opt.datatest != "":
        test_dataset = datasetPreparation(root=opt.datatest, objectsofinterest=opt.object, datasetName='test_Dataset',
                                          batch_size=opt.batchsize, keep_orientation=True, noise=opt.noise, sigma=opt.sigma,
                                          debugFolderPath=testDatasetImagePath, transform=transform, shuffle=True, saveAffAndBelImages=False)

    # get the number of devices
    numOfGPU = len(tf.config.list_physical_devices('GPU'))
    tf.print('Num GPUs Available: {}'.format(numOfGPU))
    if numOfGPU > 0:
        tf.print(tf.config.experimental.list_physical_devices('GPU'))

    # Explicitly place tensors on the DirectML device:
    # /DML:1 --> intel
    # /DML:0 --> AMD
    # /cpu --> for a cpu training
    # /device:GPU:0 --> GPU T4 on GCP
    with tf.device('/GPU:0'):
        try:
            print('train dataset number of batches: {}'.format(len(train_dataset)))
            print('test dataset number of batches: {}'.format(len(test_dataset)))

            tf.keras.backend.clear_session()

            # netModel = customModel(pretrained=opt.pretrained, blocks=6, freezeLayers=14,)
            netModel = featureModel(pretrained=opt.pretrained, blocks=6, freezeLayers=14,)
            # netModel = markoDopeModel_funcAPI(numBeliefMap=9, numAffinity=16, stop_at_stage=1, inp_shape=(400, 400, 3), pretrained=opt.pretrained)

            # model can be built by calling the build function but then all of the layers have to be used.
            # or by calling the fit function
            # to load weights model has to be built
            tf.print('building model: {}'.format(netModel.name))
            netModel.build(input_shape=(None, 400, 400, 3))

            if opt.net == '':
                tf.print('new model created')
            else:
                tf.print('loading weights from: {}'.format(opt.net))
                netModel.load_weights(filepath=opt.net)
                # netModel = tf.keras.models.load_model(filepath=opt.net)

            epochRange = range(opt.epochs)
            tf.print('epochRange is: {}'.format(epochRange))

            # info about current epoch
            epoch_count = 0

            netModel.summary()

            # plot model, if it is created it can be ploted, if it was loaded it can't?
            tf.print('ploting_model...')
            tf.keras.utils.plot_model(model=netModel.modelForPlot(),
                                      to_file=os.path.join(debugFolderPath, '{}_{}.png'.format(opt.namefile, 'Plot_Model')),
                                      show_shapes=True,
                                      show_layer_names=True)

            # get layers
            lay = netModel.layers

            # Custom loop
            # Instantiate an optimizer.
            optimizer = tf.keras.optimizers.Adam(lr=opt.lr)

            # Instantiate losses and metrics.
            loss_fn_beliefs = tf.keras.losses.MeanSquaredError(name='beliefs_loss')
            loss_fn_affinities = tf.keras.losses.MeanSquaredError(name='affinities_loss')

            beliefs_metric = tf.keras.metrics.MeanSquaredError(name='beliefs_accuracy', dtype=tf.float32)
            affinities_metric = tf.keras.metrics.MeanSquaredError(name='affinities_accuracy', dtype=tf.float32)

            # save the epoch loss, used later to print the loss in a matplotlib
            beliefLoss_perEpoch = []
            affinityLoss_perEpoch = []

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

            train_log_dir = os.path.join(debugFolderPath, 'logs', 'gradient_tape', current_time, 'train')
            test_log_dir = os.path.join(debugFolderPath, 'logs', 'gradient_tape', current_time, 'test')

            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

            # create csv files for the loss values
            if (os.path.exists(os.path.join(debugFolderPath, 'Beliefs_loss.txt')) and os.path.exists(os.path.join(debugFolderPath, 'Affinity_loss.txt'))):
                with open(os.path.join(debugFolderPath, 'Beliefs_loss.txt'), 'a') as bel_csv:
                    bel_csv.write('Epoch; Beliefs loss;\n')
                with open(os.path.join(debugFolderPath, 'Affinity_loss.txt'), 'a') as aff_csv:
                    aff_csv.write('Epoch; Affinity loss;\n')
            else:
                with open(os.path.join(debugFolderPath, 'Beliefs_loss.txt'), 'w') as bel_csv:
                    bel_csv.write('Epoch; Beliefs loss;\n')
                with open(os.path.join(debugFolderPath, 'Affinity_loss.txt'), 'w') as aff_csv:
                    aff_csv.write('Epoch; Affinity loss;\n')
            # log important outputs to log file
            if (os.path.exists(os.path.join(debugFolderPath, 'Beliefs_loss.txt'))):
                with open(os.path.join(debugFolderPath, 'logfile.txt'), 'a') as logfile:
                    logfile.write("start training: " + str(datetime.datetime.now().date()) + " " + str(datetime.datetime.now().time()) + "\n")
                    logfile.write("--------------------------------------------------------------------------------------------------------\n")
                    logfile.write("--------------------------------------------------------------------------------------------------------\n")
            else:
                with open(os.path.join(debugFolderPath, 'logfile.txt'), 'w') as logfile:
                    logfile.write("start training: " + str(datetime.datetime.now().date()) + " " + str(datetime.datetime.now().time()) + "\n")
                    logfile.write("--------------------------------------------------------------------------------------------------------\n")
                    logfile.write("--------------------------------------------------------------------------------------------------------\n")

            # check dataset
            batch_check = train_dataset[0]
            tf.print('batch_check[0][images] shape-> {}'.format(tf.shape(batch_check[0]['images'])))
            tf.print('batch_check[0][beliefs] shape-> {}'.format(tf.shape(batch_check[1]['beliefs'])))
            tf.print('batch_check[0][affinities] shape-> {}'.format(tf.shape(batch_check[1]['affinities'])))
            tf.print('')
            tf.print('batch_check[0][images] max-> {}, min-> {}'.format(tf.reduce_max(batch_check[0]['images']), tf.reduce_min(batch_check[0]['images'])))
            tf.print('batch_check[0][beliefs] max-> {}, min-> {}'.format(tf.reduce_max(batch_check[1]['beliefs']), tf.reduce_min(batch_check[1]['beliefs'])))
            tf.print('batch_check[0][affinities] max-> {}, min-> {}'.format(tf.reduce_max(batch_check[1]['affinities']), tf.reduce_min(batch_check[1]['affinities'])))

            for epoch in epochRange:
                tf.print("\nStart of epoch {}".format(epoch))
                epoch_count = epoch
                epoch_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
                # log important outputs to log file
                with open(os.path.join(debugFolderPath, 'logfile.txt'), 'a') as logfile:
                    logfile.write("Start of epoch {}".format(epoch) + "\n")

                # run the forward pass for training
                netModel, beliefs_metric, affinities_metric, _, _ = forwardPass(netModel,
                                                                                train_dataset,
                                                                                optimizer,
                                                                                loss_fn_beliefs,
                                                                                loss_fn_affinities,
                                                                                beliefs_metric,
                                                                                affinities_metric,
                                                                                epoch,
                                                                                labelsLogitsImgPath,
                                                                                training=True)

                # END OF THE EPOCH CODE
                # append to the epoch metrics
                beliefLoss_perEpoch.append(beliefs_metric.result())
                affinityLoss_perEpoch.append(affinities_metric.result())

                # tensorboard logs update
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss_aff_metrics', affinities_metric.result(), step=epoch)
                    tf.summary.scalar('loss_bel_metrics', beliefs_metric.result(), step=epoch)

                # log important output to logfile
                # with open(os.path.join(debugFolderPath, 'logfile.txt'), 'a') as logfile:
                #     logfile.write("EPOCH {} ||Training loss ---> affinity loss: {}, beliefs loss: {}".format(epoch, float(affinities_metric.result().numpy()), float(beliefs_metric.result().numpy())) + "\n")
                #     logfile.write("Seen so far: %s samples" % ((step + 1) * opt.batchsize) + "\n")
                #     logfile.write("--------------------------------------------------------------------------------------------------------\n")
                # fill list of beliefs and affinity losses
                with open(os.path.join(debugFolderPath, 'Beliefs_loss.txt'), 'a') as bel_csv:
                    bel_csv.write('{}; {};\n'.format(epoch, repr(round(float(beliefs_metric.result().numpy()), 20))))
                with open(os.path.join(debugFolderPath, 'Affinity_loss.txt'), 'a') as aff_csv:
                    aff_csv.write('{}; {};\n'.format(epoch, repr(round(float(affinities_metric.result().numpy()), 20))))

                # reset metrics per batch
                beliefs_metric.reset_states()
                affinities_metric.reset_states()

                # run the test dataset if exists
                if test_dataset is not None and opt.usetestdataset:

                    beliefs_metric_test = tf.keras.metrics.MeanSquaredError(name='beliefs_accuracy', dtype=tf.float32)
                    affinities_metric_test = tf.keras.metrics.MeanSquaredError(name='affinities_accuracy', dtype=tf.float32)

                    # run the forward pass for training
                    _, beliefs_metric_test, affinities_metric_test, beliefLoss_perBatch_test, affinityLoss_perBatch_test = forwardPass(netModel,
                                                                                                                                       test_dataset,
                                                                                                                                       optimizer,
                                                                                                                                       loss_fn_beliefs,
                                                                                                                                       loss_fn_affinities,
                                                                                                                                       beliefs_metric_test,
                                                                                                                                       affinities_metric_test,
                                                                                                                                       epoch,
                                                                                                                                       labelsLogitsImgPath,
                                                                                                                                       training=False)

                    # append to the epoch metrics
                    beliefLoss_perBatch_test.append(beliefs_metric_test.result())
                    affinityLoss_perBatch_test.append(affinities_metric_test.result())

                    # Beliefs loss
                    tf.print("SAVING BELIEFS AND AFFINITIES LOSS PLOTS...")
                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.title('Beliefs loss')
                    plt.xlabel("img nr.")
                    plt.ylabel("loss")
                    plt.plot(beliefLoss_perBatch_test)

                    # Affinities loss
                    plt.subplot(1, 2, 2)
                    plt.title('Affinities loss')
                    plt.xlabel("img nr.")
                    plt.ylabel("loss")
                    plt.plot(affinityLoss_perBatch_test)
                    plt.savefig(os.path.join(debugFolderPath, 'test_epoch_{}_affinities&beliefs_loss.png'.format(epoch)), bbox_inches='tight')

                    # reset test metrics
                    beliefs_metric_test.reset_states()
                    affinities_metric_test.reset_states()

                # save a model after each epoch
                if opt.savemodelafterepoch:

                    modelPathToSave = os.path.join(modelFolder, '{}_blocks_{}'.format(opt.namefile, netModel.blocks))
                    ckptPathToSave = os.path.join(ckptFolder, '{}_blocks_{}'.format(opt.namefile, netModel.blocks))

                    # save a model after each epoch in the 'tf' format
                    tf.print('Saving the model: {}'.format(modelPathToSave))
                    netModel.save(modelPathToSave, save_format='tf')

                    # save a checkpoints after each epoch to be able to load weights
                    tf.print('Saving the checkpoints: {}'.format(ckptPathToSave))
                    netModel.save_weights(os.path.join(ckptPathToSave, 'cp.ckpt'))

                    # calculate epoch time
                    epoch_end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
                    epoch_duration = (datetime.datetime.strptime(epoch_end_time, '%H:%M:%S') - datetime.datetime.strptime(epoch_start_time, '%H:%M:%S'))
                    tf.print('Epoch duration(h:mm:ss): {}'.format(epoch_duration))
                    # log important outputs to log file
                    with open(os.path.join(debugFolderPath, 'logfile.txt'), 'a') as logfile:
                        logfile.write('Epoch duration: {}'.format(epoch_duration) + "\n")

            tf.print("Training finished after {} epochs.".format(epoch_count+1))
            # log important outputs to log file
            with open(os.path.join(debugFolderPath, 'logfile.txt'), 'a') as logfile:
                logfile.write("Training finished after {} epochs.".format(epoch_count) + "\n")
            tf.print("end training: ", datetime.datetime.now().time())
            # log important outputs to log file
            with open(os.path.join(debugFolderPath, 'logfile.txt'), 'a') as logfile:
                logfile.write("end training: " + str(datetime.datetime.now().date()) + " " + str(datetime.datetime.now().time()) + "\n")

            # Beliefs loss
            tf.print("SAVING BELIEFS AND AFFINITIES LOSS PLOTS...")
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.title('Beliefs loss')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(beliefLoss_perEpoch)

            # Affinities loss
            plt.subplot(1, 2, 2)
            plt.title('Affinities loss')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(affinityLoss_perEpoch)
            plt.savefig(os.path.join(debugFolderPath, 'train_affinities&beliefs_loss.png'), bbox_inches='tight')

            tf.print("END TRAINING")

        except ImportError as ie:
            print('import error:\n {}'.format(ie))

        except ValueError as ve:
            print('value error:\n {}'.format(ve))

        except:
            e = sys.exc_info()[0]
            print('exception: {}'.format(e))

        print("end:", datetime.datetime.now().time())
