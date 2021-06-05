from __future__ import print_function
import sys
import os
import datetime
import cv2
import json
import numpy as np
import tensorflow as tf

# help functions
def create3DCuboidPts(x, y, z):
    """
    creates a cuboid points from the x, y and z dimensions
    """
    """
    back_up_left = (-x/2,  -y/2,  -z/2)
    back_up_right = (x/2, -y/2,  -z/2)
    back_dn_left = (-x/2,  y/2, -z/2)
    back_dn_right = (x/2, y/2, -z/2)
    front_up_right = (x/2, -y/2,  z/2)
    front_up_left = (-x/2,  -y/2,  z/2)
    front_dn_right = (x/2, y/2, z/2)
    front_dn_left = (-x/2,  y/2, z/2)
    """
    """
    cuboid_dict = {'back_up_left':     back_up_left,
                   'back_up_right':    back_up_right,
                   'back_dn_left':     back_dn_left,
                   'back_dn_right':    back_dn_right,
                   'front_up_right':   front_up_right,
                   'front_up_left':    front_up_left,
                   'front_dn_right':   front_dn_right,
                   'front_dn_left':    front_dn_left}
    """
    """ """
    back_up_left = (x/2, -y/2, -z/2)
    back_up_right = (-x/2, -y/2, -z/2)
    back_dn_left = (-x/2, y/2, -z/2)
    back_dn_right = (x/2, y/2, -z/2)
    front_up_right = (-x/2, -y/2, z/2)
    front_up_left = (x/2, -y/2, z/2)
    front_dn_right = (x/2, y/2, z/2)
    front_dn_left = (-x/2, y/2, z/2)
    
    cuboid_dict = {'back_up_left':     back_up_left,
                   'back_up_right':    back_up_right,
                   'back_dn_left':     back_dn_left,
                   'back_dn_right':    back_dn_right,
                   'front_up_left':    front_up_left,
                   'front_up_right':   front_up_right,
                   'front_dn_left':    front_dn_left,
                   'front_dn_right':   front_dn_right}

    return cuboid_dict

def getProjectedModel2DPts(model_points_dict_3D, rot_v, tran_v, matrix_camera, dist_coeffs):

    """
    return a dictionary of projected model 2D points
    """
    model_points_dict_2D = {}
    for index, point in enumerate(np.array(list(model_points_dict_3D.values()), dtype=np.float32)):
        # get the 2D projections of the model points
        (pointCoord, jacobian) = cv2.projectPoints(point, rot_v, tran_v, matrix_camera, dist_coeffs)
        # get the point coords
        pointCoord = tuple(pointCoord.astype(np.int32).ravel())

        # add the point to the 2D coords dict
        model_points_dict_2D.update({list(model_points_dict_3D.keys())[index]: pointCoord})

    return model_points_dict_2D

def getPointCoord(pointTensor):
    """
    from the tensor layer holding one filter(image), get the 2D coordinate of the point in the image
    """

    # flatten the tensor
    featureFlatten = tf.reshape(pointTensor, [-1])

    # get the max argument of the flatten array
    featureFlattenArgMax = tf.math.argmax(featureFlatten)

    # get the coordinates of that argument in tensor of the shape as maskImageTensor
    coords = tf.unravel_index(indices=tf.cast(featureFlattenArgMax, dtype=tf.int32), dims=tf.cast(tf.shape(pointTensor), dtype=tf.int32))
    coords = tf.roll(coords, shift=1, axis=0)   # roll coords to get the right width height format

    return coords.numpy()

def getImage(dataset):
    """
    get the image the input tensor(dataset)
    """
    tensorImg = dataset[0][0]['images']
    if len(tf.shape(tensorImg)) > 3:
        tensorLabel = tf.squeeze(tensorImg, axis=0)
    tensorImgNp = tensorLabel.numpy()
    tensorImg = cv2.cvtColor(tensorImgNp.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return tensorImg

def getBatchImage(batch):
    """
    get the image the input tensor(dataset)
    """
    tensorImg = batch
    if len(tf.shape(tensorImg)) > 3:
        tensorImg = tf.squeeze(tensorImg, axis=0)
    tensorImgNp = tensorImg.numpy()
    tensorImg = cv2.cvtColor(tensorImgNp.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return tensorImg

def getImagePointCoords(tensor):
    """
    Function to get the list of pixel coordinates for each point in the tensor layer
    using the getPointCoord
    """
    dimTensor = len(tf.shape(tensor))

    # if dim is bigger than 3, it means the first axis is a batch and
    # it should be removed because the logit will not have 4 dimm
    if dimTensor > 3:
        tensor = tf.squeeze(tensor, axis=0)

    size = tf.shape(tensor)

    # number of points(filters)
    numOfFilters = tf.shape(tensor[0])[-1]

    # fill the dict with coords of the each 2D point
    image_points = {}
    for index, point in enumerate(range(numOfFilters)):
        image_points[index] = getPointCoord(tensor[:, :, point])

    if numOfFilters == 9:   # Belief
        # assign coords derived from filer to vertices of the cuboid
        image_points_dict = {'back_up_left':     image_points[5],
                             'back_up_right':    image_points[4],
                             'back_dn_left':     image_points[7],
                             'back_dn_right':    image_points[6],
                             'front_up_left':    image_points[1],
                             'front_up_right':   image_points[0],
                             'front_dn_left':    image_points[3],
                             'front_dn_right':   image_points[2], }

    if numOfFilters > 9:    # Affinity
        # assign coords derived from filer to vertices of the cuboid
        image_points_dict = {'back_up_left':     image_points[5],
                             'back_up_right':    image_points[4],
                             'back_dn_left':     image_points[7],
                             'back_dn_right':    image_points[6],
                             'front_up_left':    image_points[1],
                             'front_up_right':   image_points[0],
                             'front_dn_left':    image_points[3],
                             'front_dn_right':   image_points[2], }

    image_points = np.array(list(image_points_dict.values()))

    return image_points, image_points_dict, size

def addCoordOnImage(image, image_points, scale, color):
    """
    draws a point of color in the position given in image points on the given image 
    """

    image = np.copy(image)

    # copy the image first, otherwise it will draw on the input image
    ImgFrom_image_points = np.copy(image)

    #convert to uint8
    image_points = image_points.astype(np.uint8)

    for index, point in enumerate(image_points):
        cv2.circle(img=ImgFrom_image_points, center=tuple(point*scale), radius=2, color=color, thickness=-1)
        cv2.putText(ImgFrom_image_points,
                    # '{}, {} {}'.format(index, point[1]*scale, point[0]*scale),
                    '{}'.format(index),
                    tuple(point*scale),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)

    return ImgFrom_image_points

def addTextOnImage(image, text, text_width_ratio, text_height_ratio, color):

    # copy the image first, otherwise it will draw on the input image
    ImgFrom_image_points = np.copy(image)

    cv2.putText(ImgFrom_image_points, '{}'.format(text),
            #(int(ImgFrom_image_points.shape[1]-ImgFrom_image_points.shape[0]/2), int(0+ImgFrom_image_points.shape[1]/15)),  # upper rigth corner
            (int(ImgFrom_image_points.shape[0]*text_width_ratio), int(ImgFrom_image_points.shape[1]*text_height_ratio)),  # upper rigth corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)

    return ImgFrom_image_points

def removeZeroCoord(image_points_dict, model_points_dict):

    # get keys with the zero coord values
    zero_coord_keys = [key for key, val in image_points_dict.items() if (val == np.zeros(shape=2, dtype=np.float32)).all()]

    image_points_dict = {key: value for key, value in image_points_dict.items() if key not in zero_coord_keys}
    model_points_dict = {key: value for key, value in model_points_dict.items() if key not in zero_coord_keys}

    return image_points_dict, model_points_dict

def draw_axis(image, rot, tran, mat_cam, dist_coeffs):
    # unit is mm
    rot, _ = cv2.Rodrigues(rot)
    # points [x,y,z]

    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, jac = cv2.projectPoints(points, rot, tran, mat_cam, dist_coeffs)

    #convert to uint8
    axisPoints = axisPoints.astype(np.uint8)

    # copy the image first, otherwise it will draw on the input image
    image = np.copy(image)
    #color conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # x color red
    image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 2)
    # y color green
    image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 2)
    # z color blue
    image = cv2.line(image, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 2)

    return image

def draw_cuboid(image, model_points_2D, color):

    image_cuboid = np.copy(image)
    """
    image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_left'], model_points_2D['back_up_right'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_left'], model_points_2D['back_dn_left'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_left'], model_points_2D['front_up_left'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_right'], model_points_2D['back_dn_right'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_right'], model_points_2D['front_up_right'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['front_up_right'], model_points_2D['front_dn_right'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['front_up_right'], model_points_2D['front_up_left'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['front_up_left'], model_points_2D['front_dn_left'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['front_dn_left'], model_points_2D['front_dn_right'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['front_dn_left'], model_points_2D['back_dn_left'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['front_dn_right'], model_points_2D['back_dn_right'], (0, 0, 255), 1)
    image_cuboid = cv2.line(image_cuboid, model_points_2D['back_dn_right'], model_points_2D['back_dn_left'], (0, 0, 255), 1)
    """

    if (('back_up_left' in model_points_2D) and ('back_up_right' in model_points_2D)):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_left'], model_points_2D['back_up_right'], color, 1)
    if (('back_up_right' in model_points_2D) and ('back_dn_left' in model_points_2D)):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_right'], model_points_2D['back_dn_left'], color, 1)
    if (('back_up_left' in model_points_2D) and ('front_up_left' in model_points_2D)):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_left'], model_points_2D['front_up_left'], color, 1)
    if (('back_up_left' in model_points_2D) and ('back_dn_right' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_left'], model_points_2D['back_dn_right'], color, 1)
    if (('back_up_right' in model_points_2D) and ('front_up_right' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['back_up_right'], model_points_2D['front_up_right'], color, 1)
    if (('front_up_left' in model_points_2D) and ('front_dn_right' in model_points_2D) ): 
        image_cuboid = cv2.line(image_cuboid, model_points_2D['front_up_left'], model_points_2D['front_dn_right'], color, 1)
    if (('front_up_right' in model_points_2D) and ('front_up_left' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['front_up_right'], model_points_2D['front_up_left'], color, 1)
    if (('front_up_right' in model_points_2D) and ('front_dn_left' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['front_up_right'], model_points_2D['front_dn_left'], color, 1)
    if (('front_dn_left' in model_points_2D) and ('front_dn_right' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['front_dn_left'], model_points_2D['front_dn_right'], color, 1)
    if (('front_dn_left' in model_points_2D) and ('back_dn_left' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['front_dn_left'], model_points_2D['back_dn_left'], color, 1)
    if (('front_dn_right' in model_points_2D) and ('back_dn_right' in model_points_2D) ): 
        image_cuboid = cv2.line(image_cuboid, model_points_2D['front_dn_right'], model_points_2D['back_dn_right'], color, 1)
    if (('back_dn_right' in model_points_2D) and ('back_dn_left' in model_points_2D) ):
        image_cuboid = cv2.line(image_cuboid, model_points_2D['back_dn_right'], model_points_2D['back_dn_left'], color, 1)

    return image_cuboid


class positionSolver():

    def __init__(self,
                 camSettingsJSON,
                 cameraSettings_cv_path,
                 use_opencv_intrinsics,
                 objSettingsJSON,
                 debug,
                 text_width_ratio,
                 text_height_ratio,
                 text='',
                 belColor=(0, 0, 255),
                 affColor=(0, 255, 255)):
        
        self.camSettingsJSON = camSettingsJSON
        self.cameraSettings_cv_path = cameraSettings_cv_path
        self.use_opencv_intrinsics = use_opencv_intrinsics
        self.objSettingsJSON = objSettingsJSON
        self.belColor = belColor
        self.affColor = affColor
        self.text_width_ratio = text_width_ratio
        self.text_height_ratio = text_height_ratio
        self.text = text
        self.debug = debug
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # camera settings from ndds
        if not self.use_opencv_intrinsics:
            # read camera json file
            camera_matrix_path = self.camSettingsJSON
            with open(camera_matrix_path) as data_file:
                json_camera_matrix = json.load(data_file)

            # Initialize camera parameters
            self.matrix_camera = np.zeros((3, 3))
            self.matrix_camera[0, 0] = json_camera_matrix['camera_settings'][0]['intrinsic_settings']['fx']
            self.matrix_camera[1, 1] = json_camera_matrix['camera_settings'][0]['intrinsic_settings']['fy']
            self.matrix_camera[0, 2] = json_camera_matrix['camera_settings'][0]['intrinsic_settings']['cx']
            self.matrix_camera[1, 2] = json_camera_matrix['camera_settings'][0]['intrinsic_settings']['cy']
            self.matrix_camera[2, 2] = 1
        
        # camera settings from opencv
        if self.use_opencv_intrinsics:
            self.CamSettings = cv2.FileStorage(self.cameraSettings_cv_path, cv2.FILE_STORAGE_READ)
            self.matrix_camera = self.CamSettings.getNode("intrinsic_settings").mat()
            self.dist_coeffs = self.CamSettings.getNode("dist").mat()

        #Fixed model transform
        object_sett_path = self.objSettingsJSON
        with open(object_sett_path) as data_file:
            object_sett_matrix = json.load(data_file)

        # Initialize parameters
        fixed_model_transform = np.array(object_sett_matrix['exported_objects'][0]['fixed_model_transform'])
        print('fixed model transform: \n{}\n'.format(fixed_model_transform))

        # 3D model points. Dimension of the cube around the model (in mm???)
        # should be 16(affinity) or 9(belief) points
        self.obj_dim = object_sett_matrix['exported_objects'][0]['cuboid_dimensions']         # --> cm??
        self.model_points = create3DCuboidPts(x=self.obj_dim[0]*10, y=self.obj_dim[1]*10, z=self.obj_dim[2]*10)  # mm ??
        print('model cuboid points:')
        {print('\t{}:{}'.format(key, value)) for key, value in self.model_points.items()}

    def getPosition(self, belTensor, affTensor, tensor_image):
        #img = np.copy(image)
        tensor_img = tensor_image

        bel_image_points, bel_image_points_dict, _, = getImagePointCoords(belTensor)

        # print('belief image points list: \n {}\n'.format(bel_image_points))
        #print('belief image points dict:')
        #{print('\t{}:{}'.format(key, value)) for key, value in bel_image_points_dict.items()}
        bel_shape = tf.shape(belTensor)[1:3]
        bel_scale = int(tf.shape(tensor_img)[1]/tf.shape(belTensor)[1])

        ############################################################################################################
        # AFFINITY get the point coordinates from each filter(point), these will be our image_points
        aff_image_points, aff_image_points_dict, _, = getImagePointCoords(affTensor)
        aff_shape = tf.shape(affTensor)[1:3]
        aff_scale = int(tf.shape(tensor_img)[1]/tf.shape(affTensor)[1])

        # check if there is a point at pixel coordinate is (0, 0) and remove that point
        # (0, 0) not all points are in the image and when we create an image from labels it will have coordinates (0,0)
        # for example if in belief maps the filter is completely black
        bel_image_points_dict, model_points_dict_3D = removeZeroCoord(image_points_dict=bel_image_points_dict, model_points_dict=self.model_points)
        
        belPointsOnImage = addCoordOnImage(image=tensor_img, image_points=bel_image_points, scale=bel_scale, color=self.belColor)
        affPointsOnImage = addCoordOnImage(image=tensor_img, image_points=aff_image_points, scale=aff_scale, color=self.affColor)

        #belPointsOnImage = addCoordOnImage(image=img, image_points=bel_image_points, imgName='beliefs', scale=bel_scale, color=self.belColor)
        #affPointsOnImage = addCoordOnImage(image=img, image_points=aff_image_points, imgName='affinity', scale=aff_scale, color=self.affColor)

        (success, rot_v, tran_v) = cv2.solvePnP(objectPoints=np.array(list(model_points_dict_3D.values()), dtype=np.float32),
                                                # imagePoints=(np.array(list(aff_image_points_dict.values()), dtype=np.float32) * aff_scale),
                                                imagePoints=(np.array(list(bel_image_points_dict.values()), dtype=np.float32) * bel_scale),
                                                cameraMatrix=self.matrix_camera,
                                                distCoeffs=self.dist_coeffs,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            #print('Rotation Vector:\n {}\n'.format(rot_v))
            #print('Translation Vector:\n {}\n'.format(tran_v))

            x = tran_v[0]
            y = tran_v[1]
            z = tran_v[2]

            #if z < 0:
                # Get the opposite location
                #tran_v = np.array([-x, -y, -z])

                # # Change the rotation by 180 degree
                # rotate_angle = np.pi
                # rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                # quaternion = rotate_quaternion.cross(quaternion)

            # get the image from the dataset
            #tensorImg = getImage(test_dataset)
            tensorImg = getBatchImage(tensor_img)
            # draw axis
            if self.debug:
                tensorImg = draw_axis(tensorImg, rot_v, tran_v, self.matrix_camera, self.dist_coeffs)

                # get model points projections on the image
                model_points_dict_2D = getProjectedModel2DPts(model_points_dict_3D, rot_v, tran_v, self.matrix_camera, self.dist_coeffs)

                # draw point projections on the image
                #projPts_color = (0, 0, 255)     # (BGR) red
                tensorImg = addCoordOnImage(image=tensorImg,
                                            image_points=np.array(list(model_points_dict_2D.values()), dtype=np.float32),
                                            #imgName= self.text,
                                            scale=1, color=self.belColor)

                tensorImg = addTextOnImage(image=tensorImg, text=self.text, text_width_ratio = self.text_width_ratio, 
                                           text_height_ratio=self.text_height_ratio, color=self.belColor)
                # draw a cuboid around the object (connect points)
                tensorImg = draw_cuboid(tensorImg, model_points_dict_2D, color=self.belColor)

                if (belPointsOnImage.shape[0]==1):
                    belPointsOnImage = np.squeeze(belPointsOnImage, axis=0)
                belPointsOnImage = belPointsOnImage.astype(np.uint8)

                if (tensorImg.shape[0]==1):
                    tensorImg = np.squeeze(tensorImg, axis=0)
                tensorImg = tensorImg.astype(np.uint8)

                # Display projected points on image and image points
                #img_points_projected_points = cv2.hconcat([belPointsOnImage, tensorImg])
                img_points_projected_points = cv2.hconcat([belPointsOnImage, tensorImg])
                #cv2.imshow('img_points_projected_points', img_points_projected_points)
                #cv2.waitKey(0)

        else:
            print("position unresolved \n")

        return rot_v, tran_v, tensorImg, img_points_projected_points