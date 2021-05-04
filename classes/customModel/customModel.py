import sys
import os

import tensorflow as tf

class customModel(tf.keras.Model):
    # number of stages to process (if less than total number of stages)
    def __init__(self,
                 pretrained=False,
                 numBeliefMap=9,
                 numAffinity=16,
                 blocks=6,
                 kerasInit=tf.keras.initializers.GlorotNormal(),  # tf.keras.initializers.glorot_normal(),
                 biasInit=tf.keras.initializers.Zeros(),
                 freezeLayers=10,
                 inp_shape=(400, 400, 3)):

        # call the __init__ of the base class (nn.Module)
        super().__init__()

        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.blocks = blocks
        self.kerasInit = kerasInit
        self.biasInit = biasInit
        self.freezeLayers = freezeLayers
        self.inp_shape = inp_shape

        if self.pretrained is False:
            print("Training network without imagenet weights.")
            vgg_full = tf.keras.applications.VGG19(include_top=False, weights=None, input_shape=self.inp_shape)  # input_shape=(400, 400, 3) ??
        else:
            print("Training network pretrained on imagenet.")
            vgg_full = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=self.inp_shape)  # input_shape=(400, 400, 3) ??

        # DESIGN NETWORK MODEL FROM THE keras.models.vgg19:
        self.layer00 = vgg_full.layers[0]   # input layer 
        # [400, 400, 3]
        self.layer01 = vgg_full.layers[1]   # conv(kernel=(3,3), filters=64)
        self.layer02 = vgg_full.layers[2]   # conv(kernel=(3,3), filters=64)
        self.layer03 = vgg_full.layers[3]   # MaxPool
        # [200, 200, 3]
        self.layer04 = vgg_full.layers[4]   # conv(kernel=(3,3), filters=128)
        self.layer05 = vgg_full.layers[5]   # conv(kernel=(3,3), filters=128)
        self.layer06 = vgg_full.layers[6]   # MaxPool
        # [100, 100, 3]
        self.layer07 = vgg_full.layers[7]   # conv(kernel=(3,3), filters=256)
        self.layer08 = vgg_full.layers[8]   # conv(kernel=(3,3), filters=256)
        self.layer09 = vgg_full.layers[9]   # conv(kernel=(3,3), filters=256)
        self.layer10 = vgg_full.layers[10]  # conv(kernel=(3,3), filters=256)

        # delete the vgg_full, it is not needed anymore, we will continue on the self.layers
        del vgg_full

        # vgg19 will have maxpooling as last layer, this will replace it with additional conv2D layer
        self.layer11 = tf.keras.layers.Dropout(rate=0.2, name="lay11_dropout")
        self.layer12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay12_Conv2D", kernel_initializer=kerasInit)
        self.layer13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay13_Conv2D", kernel_initializer=kerasInit)
        self.layer14 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay14_Conv2D", kernel_initializer=kerasInit)
        self.layer15 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay15_Conv2D", kernel_initializer=kerasInit)
        self.layer16 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay16_Conv2D", kernel_initializer=kerasInit)
        self.layer17 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay17_Conv2D", kernel_initializer=kerasInit)
    
        self.layer18 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay18_Conv2D", kernel_initializer=kerasInit)
        self.layer19 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name="lay19_Conv2D", kernel_initializer=kerasInit)
        self.layer20 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="lay20_MaxPool2D")
        self.layer21 = tf.keras.layers.BatchNormalization(name="BatchNorm1")

        if self.blocks == 1:
            self.belief_1 = self.create_block(512, numBeliefMap, first=True, name='belief1')

            self.affinity_1 = self.create_block(512, numAffinity, first=True, name='affinity1')

        if self.blocks == 2:
            self.belief_1 = self.create_block(512, numBeliefMap, first=True, name='belief1')
            self.belief_2 = self.create_block(512 + 1*numBeliefMap, numBeliefMap, first=False, name='belief2')

            self.affinity_1 = self.create_block(512, numAffinity, first=True, name='affinity1')
            self.affinity_2 = self.create_block(512 + 1*numAffinity, numAffinity, first=False, name='affinity2')

            self.layer22 = tf.keras.layers.BatchNormalization(name="BatchNorm2")

        if self.blocks == 3:
            self.belief_1 = self.create_block(512, numBeliefMap, first=True, name='belief1')
            self.belief_2 = self.create_block(512 + 1*numBeliefMap, numBeliefMap, first=False, name='belief2')
            self.belief_3 = self.create_block(512 + 2*numBeliefMap, numBeliefMap, first=False, name='belief3')

            self.affinity_1 = self.create_block(512, numAffinity, first=True, name='affinity1')
            self.affinity_2 = self.create_block(512 + 1*numAffinity, numAffinity, first=False, name='affinity2')
            self.affinity_3 = self.create_block(512 + 2*numAffinity, numAffinity, first=False, name='affinity3')
            self.layer22 = tf.keras.layers.BatchNormalization(name="BatchNorm2")
            self.layer23 = tf.keras.layers.BatchNormalization(name="BatchNorm3")

        if self.blocks == 4:
            self.belief_1 = self.create_block(512, numBeliefMap, first=True, name='belief1')
            self.belief_2 = self.create_block(512 + 1*numBeliefMap, numBeliefMap, first=False, name='belief2')
            self.belief_3 = self.create_block(512 + 2*numBeliefMap, numBeliefMap, first=False, name='belief3')
            self.belief_4 = self.create_block(512 + 3*numBeliefMap, numBeliefMap, first=False, name='belief4')

            self.affinity_1 = self.create_block(512, numAffinity, first=True, name='affinity1')
            self.affinity_2 = self.create_block(512 + 1*numAffinity, numAffinity, first=False, name='affinity2')
            self.affinity_3 = self.create_block(512 + 2*numAffinity, numAffinity, first=False, name='affinity3')
            self.affinity_4 = self.create_block(512 + 3*numAffinity, numAffinity, first=False, name='affinity4')

            self.layer22 = tf.keras.layers.BatchNormalization(name="BatchNorm2")
            self.layer23 = tf.keras.layers.BatchNormalization(name="BatchNorm3")
            self.layer24 = tf.keras.layers.BatchNormalization(name="BatchNorm4")

        if self.blocks == 5:
            self.belief_1 = self.create_block(512, numBeliefMap, first=True, name='belief1')
            self.belief_2 = self.create_block(512 + 1*numBeliefMap, numBeliefMap, first=False, name='belief2')
            self.belief_3 = self.create_block(512 + 2*numBeliefMap, numBeliefMap, first=False, name='belief3')
            self.belief_4 = self.create_block(512 + 3*numBeliefMap, numBeliefMap, first=False, name='belief4')
            self.belief_5 = self.create_block(512 + 4*numBeliefMap, numBeliefMap, first=False, name='belief5')

            self.affinity_1 = self.create_block(512, numAffinity, first=True, name='affinity1')
            self.affinity_2 = self.create_block(512 + 1*numAffinity, numAffinity, first=False, name='affinity2')
            self.affinity_3 = self.create_block(512 + 2*numAffinity, numAffinity, first=False, name='affinity3')
            self.affinity_4 = self.create_block(512 + 3*numAffinity, numAffinity, first=False, name='affinity4')
            self.affinity_5 = self.create_block(512 + 4*numAffinity, numAffinity, first=False, name='affinity5')

            self.layer22 = tf.keras.layers.BatchNormalization(name="BatchNorm2")
            self.layer23 = tf.keras.layers.BatchNormalization(name="BatchNorm3")
            self.layer24 = tf.keras.layers.BatchNormalization(name="BatchNorm4")
            self.layer25 = tf.keras.layers.BatchNormalization(name="BatchNorm5")

        if self.blocks == 6:
            self.belief_1 = self.create_block(512, numBeliefMap, first=True, name='belief1')
            self.belief_2 = self.create_block(512 + 1*numBeliefMap, numBeliefMap, first=False, name='belief2')
            self.belief_3 = self.create_block(512 + 2*numBeliefMap, numBeliefMap, first=False, name='belief3')
            self.belief_4 = self.create_block(512 + 3*numBeliefMap, numBeliefMap, first=False, name='belief4')
            self.belief_5 = self.create_block(512 + 4*numBeliefMap, numBeliefMap, first=False, name='belief5')
            self.belief_6 = self.create_block(512 + 5*numBeliefMap, numBeliefMap, first=False, name='belief6')

            self.affinity_1 = self.create_block(512, numAffinity, first=True, name='affinity1')
            self.affinity_2 = self.create_block(512 + 1*numAffinity, numAffinity, first=False, name='affinity2')
            self.affinity_3 = self.create_block(512 + 2*numAffinity, numAffinity, first=False, name='affinity3')
            self.affinity_4 = self.create_block(512 + 3*numAffinity, numAffinity, first=False, name='affinity4')
            self.affinity_5 = self.create_block(512 + 4*numAffinity, numAffinity, first=False, name='affinity5')
            self.affinity_6 = self.create_block(512 + 5*numAffinity, numAffinity, first=False, name='affinity6')

            self.layer22 = tf.keras.layers.BatchNormalization(name="BatchNorm22")
            self.layer23 = tf.keras.layers.BatchNormalization(name="BatchNorm23")
            self.layer24 = tf.keras.layers.BatchNormalization(name="BatchNorm24")
            self.layer25 = tf.keras.layers.BatchNormalization(name="BatchNorm25")
            self.layer26 = tf.keras.layers.BatchNormalization(name="BatchNorm26")

            self.layer32 = tf.keras.layers.BatchNormalization(name="BatchNorm32")
            self.layer33 = tf.keras.layers.BatchNormalization(name="BatchNorm33")
            self.layer34 = tf.keras.layers.BatchNormalization(name="BatchNorm34")
            self.layer35 = tf.keras.layers.BatchNormalization(name="BatchNorm35")
            self.layer36 = tf.keras.layers.BatchNormalization(name="BatchNorm36")

        # freeze layers
        for layer in range(self.freezeLayers):
            self.layers[layer].trainable = False

    def call(self, inputs):
        ''' Runs inference on the neural network
            inputs is a dictionary consisted of keys ['img':tensor, 'belief':tensor, 'affinity':tensor]
            Pushes the inputs(batch) input through the certain layers(conv2D and MaxPooling) after the last layer it will
            the out1 will be of shape shape(None, 50, 50, 128) which means (batch_size, imageDimension1, imageDimension2, Filters)
            MaxPooling layers are used to decrease dimensions to be imageDimension1=50, imageDimension2=50

            Further more, out1 layer is pushed through a belief and affnity models for 6 blocks(unless defined differently on declaration)
            and returns the list of the belief and affinities tensors
        '''
        # inputs = tf.keras.layers.Input(self.inp_shape, dtype=tf.float32)
        # x = tf.cast(inputs, tf.float32)
        x = tf.keras.applications.vgg19.preprocess_input(inputs)  # (x)
        x = self.layer00(x)
        x = self.layer01(x)
        x = self.layer02(x)
        x = self.layer03(x)
        x = self.layer04(x)
        x = self.layer05(x)
        x = self.layer06(x)
        x = self.layer07(x)
        x = self.layer08(x)
        x = self.layer09(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        self.out = self.layer21(x)

        self.bel1 = self.belief_1(self.out)    # shape(None, 50, 50, 9)    Belief
        self.aff1 = self.affinity_1(self.out)    # shape(None, 50, 50, 16)   Affinities

        if self.blocks == 1:
            # return tf.keras.Model(inputs=firstLayer, outputs=[out1_2, out1_1])
            return self.bel1, self.aff1

        # concatonate on axis=3 means it will make another tensor of the shape (None, 50, 50, (128+9+16))
        x1 = tf.concat(values=[self.bel1, self.out], axis=3)
        x2 = tf.concat(values=[self.aff1, self.out], axis=3)
        # x = tf.concat(values=[self.bel1, self.aff1, self.out], axis=3)
        # self.out2 = self.layer22(x)
        x1 = self.layer22(x1)
        x2 = self.layer32(x2)

        self.bel2 = self.belief_2(x1)
        self.aff2 = self.affinity_2(x2)

        if self.blocks == 2:
            # return tf.keras.Model(inputs=firstLayer, outputs=[out2_2, out2_1])
            return self.bel2, self.aff2

        # concatonate on axis=3 means it will make another tensor of the shape (None, 50, 50, (128+9+16))
        x1 = tf.concat(values=[self.bel2, x1], axis=3)
        x2 = tf.concat(values=[self.aff2, x2], axis=3)
        # x = tf.concat(values=[self.bel2, self.aff2, self.out], axis=3)
        # x = tf.concat(values=[self.bel2, self.aff2, self.out2], axis=3) 
        # self.out3 = self.layer23(x)   
        x1 = self.layer23(x1)
        x2 = self.layer33(x2)

        self.bel3 = self.belief_3(x1)
        self.aff3 = self.affinity_3(x2)

        if self.blocks == 3:
            # return tf.keras.Model(inputs=firstLayer, outputs=[out3_2, out3_1])
            return self.bel3, self.aff3

        # concatonate on axis=3 means it will make another tensor of the shape (None, 50, 50, (128+9+16))
        x1 = tf.concat(values=[self.bel3, x1], axis=3)
        x2 = tf.concat(values=[self.aff3, x2], axis=3)
        # x = tf.concat(values=[self.bel3, self.aff3, self.out], axis=3)   
        # x = tf.concat(values=[self.bel3, self.aff3, self.out3], axis=3)
        # self.out4 = self.layer24(x)
        x1 = self.layer24(x1)
        x2 = self.layer34(x2)

        self.bel4 = self.belief_4(x1)    # shape(None, 50, 50, 9)    Belief
        self.aff4 = self.affinity_4(x2)    # shape(None, 50, 50, 16)   Affinities

        if self.blocks == 4:
            # return tf.keras.Model(inputs=firstLayer, outputs=[out4_2, out4_1])
            return self.beli4, self.aff4

        # concatonate on axis=3 means it will make another tensor of the shape (None, 50, 50, (128+9+16))
        x1 = tf.concat(values=[self.bel4, x1], axis=3)
        x2 = tf.concat(values=[self.aff4, x2], axis=3)
        # x = tf.concat(values=[self.bel4, self.aff4, self.out], axis=3)
        # x = tf.concat(values=[self.bel4, self.aff4, self.out4], axis=3)
        # self.out5 = self.layer25(x)
        x1 = self.layer25(x1)
        x2 = self.layer35(x2)

        self.bel5 = self.belief_5(x1)    # shape(None, 50, 50, 9)    Belief
        self.aff5 = self.affinity_5(x2)    # shape(None, 50, 50, 16)   Affinities

        if self.blocks == 5:
            # return tf.keras.Model(inputs=firstLayer, outputs=[out5_2, out5_1])
            return self.bel5, self.aff5

        # concatonate on axis=3 means it will make another tensor of the shape (None, 50, 50, (128+9+16))
        x1 = tf.concat(values=[self.bel5, x1], axis=3)
        x2 = tf.concat(values=[self.aff5, x2], axis=3)
        # x = tf.concat(values=[self.bel5, self.aff5, self.out], axis=3)
        # x = tf.concat(values=[self.bel5, self.aff5, self.out5], axis=3)
        # self.out6 = self.layer26(x)
        x1 = self.layer26(x1)
        x2 = self.layer36(x2)

        self.bel6 = self.belief_6(x1)    # shape(None, 50, 50, 9)    Belief
        self.aff6 = self.affinity_6(x2)    # shape(None, 50, 50, 16)   Affinities

        # return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2], [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
        # return tf.keras.Model(inputs=firstLayer, outputs=[out6_2, out6_1])
        return self.bel6, self.aff6

    def create_block(self, in_channels, out_channels, first=False, name=''):
        ''' 
            Create networks models for each submodel
        '''
        model = tf.keras.models.Sequential(name=name)
        mid_channels = 128

        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        # model.add_module("0", nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
        # input_shape (50, 50, in_channels) -->50, 50 because the out1 is of this shape
        model.add(tf.keras.layers.Conv2D(input_shape=(50, 50, in_channels),
                                          data_format="channels_last", filters=mid_channels,
                                          kernel_size=(kernel, kernel), strides=(1, 1),
                                          padding="same", activation="relu", kernel_initializer=self.kerasInit))

        # Middle convolutions
        i = 1
        while i < count-1:
            # model.add_module(str(i), nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
            model.add(tf.keras.layers.Conv2D(filters=mid_channels, 
                                              kernel_size=(kernel, kernel), 
                                              strides=(1, 1), padding="same", 
                                              activation="relu",
                                              kernel_initializer=self.kerasInit))
            i += 1

        # Penultimate convolution
        # model.add_module(str(i), nn.Conv2d(in_channels=mid_channels, out_channels=final_channels, kernel_size=1, stride=1))
        model.add(tf.keras.layers.Conv2D(filters=final_channels, 
                                          kernel_size=(kernel, kernel), 
                                          strides=(1, 1), 
                                          padding="same", 
                                          activation="relu", 
                                          kernel_initializer=self.kerasInit))

        # Last convolution
        # model.add_module(str(i), nn.Conv2d(in_channels=final_channels, out_channels=out_channels, kernel_size=1, stride=1))
        model.add(tf.keras.layers.Conv2D(filters=out_channels,
                                          kernel_size=(kernel, kernel),
                                          strides=(1, 1),
                                          padding="same",
                                          activation="relu",
                                          kernel_initializer=self.kerasInit))

        return model

    def modelForPlot(self):
        x = tf.keras.layers.Input(shape=self.inp_shape)
        model = tf.keras.Model(inputs=[x], outputs=[self.call(x)])
        return model

