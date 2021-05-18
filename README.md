# This project is made out of curiosity and solely for non-profit purposes
	
# Models are created in Tensorflow 2.4.1 and trained on GCP platform using Ubuntu 18.04 equiped with NVIDIA V100 GPU.

# Results:
![alt text][sample1]
![alt text][sample2]
![alt text][sample3]
![alt text][sample4]

[sample1]: https://github.com/bajloml/tf_train_6DOF/blob/master/test_scripts/images/test_0.png "sample1"
[sample2]: https://github.com/bajloml/tf_train_6DOF/blob/master/test_scripts/images/test_9.png "sample2"
[sample3]: https://github.com/bajloml/tf_train_6DOF/blob/master/test_scripts/images/test_16.png "sample3"
[sample4]: https://github.com/bajloml/tf_train_6DOF/blob/master/test_scripts/images/test_13.png "sample4"

# Project is based on the 'Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects'

@inproceedings{tremblay2018corl:dope,
    author = {Jonathan Tremblay and Thang To and Balakumar Sundaralingam and Yu Xiang and Dieter Fox and Stan Birchfield},
    title = {Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects},
    booktitle = {Conference on Robot Learning (CoRL)},
    url = "https://arxiv.org/abs/1809.10790",
    year = 2018
    }

# Dataset_Synthesizer

## Dataset is created using the NDDS and UE4 engine on the model created in Blender. 
Blender model is downloaded from https://www.cgtrader.com/free-3d-models/blender

Quotation of the Synthesizer which was used to create dataset
@misc{to2018ndds,
    author = {Thang To and Jonathan Tremblay and Duncan McKay and Yukie Yamaguchi and Kirby Leung 
            and Adrian Balanon and Jia Cheng and William Hodge and Stan Birchfield},
    url = "https://github.com/NVIDIA/Dataset_Synthesizer",
    title = {{NDDS}: {NVIDIA} Deep Learning Dataset Synthesizer},
    Year = 2018
    }
    
