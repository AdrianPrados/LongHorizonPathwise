import os
import math
import pybullet as p
import time
import numpy as np
from torchvision.transforms import ToPILImage
import torch 
import torchvision.transforms as T
import torchvision.models as models
import copy
from os.path import expanduser
import omegaconf
import hydra
import gdown
from torch.hub import load_state_dict_from_url
from gymnasium import spaces
import gymnasium
import matplotlib.pyplot as plt
from PIL import Image

VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "num_negatives"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "vip.VIP"
    config["device"] = device

    return config.agent

def load_vip(modelid='resnet50'):
    home = os.path.join(expanduser("~"), ".vip")

    if not os.path.exists(os.path.join(home, modelid)):
        os.makedirs(os.path.join(home, modelid))
    folderpath = os.path.join(home, modelid)
    modelpath = os.path.join(home, modelid, "model.pt")
    configpath = os.path.join(home, modelid, "config.yaml")

    try:
        if modelid == "resnet50":
            modelurl= "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
            configurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/config.yaml"
        else:
            raise NameError('Invalid Model ID')
        if not os.path.exists(modelpath):
            load_state_dict_from_url(modelurl, folderpath)
            load_state_dict_from_url(configurl, folderpath)
    except: 
        if modelid == "resnet50":
            modelurl = 'https://drive.google.com/uc?id=1LuCFIV44xTZ0GLmLwk36BRsr9KjCW_yj'
            configurl = 'https://drive.google.com/uc?id=1XSQE0gYm-djgueo8vwcNgAiYjwS43EG-'
        else:
            raise NameError('Invalid Model ID')
        if not os.path.exists(modelpath):
            gdown.download(modelurl, modelpath, quiet=False)
        if not os.path.exists(configpath):
            gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    vip_state_dict = torch.load(modelpath, map_location=torch.device(device))['vip']
    rep.load_state_dict(vip_state_dict)
    return rep    

model = load_vip()
model = model.to(device)
transform = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
SAMPLING_RATE = 1e-6  # 1000Hz sampling rate

class PandaRobot(gymnasium.Env):
    """"""

    def __init__(self, goal_image=None, include_gripper=True, max_steps = 100, sample_freq=16, time_embedding=False):
        """"""

        super(PandaRobot, self).__init__()
        self.goal_image = self.vip_image_transformation(goal_image)
        self.time_embedding = time_embedding
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=float)
        if self.time_embedding == True:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=float)  
        self.action_space = spaces.Box(low=-4, high=4, shape=(9,), dtype=float)
        p.setAdditionalSearchPath(os.path.dirname(__file__) + '/model_description')
        panda_model = "black_panda.urdf" if include_gripper else "panda.urdf"

        self.robot_id = p.loadURDF(panda_model, [0.62,0.62,0.61], [0,0,1,1],useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        p.changeDynamics(self.robot_id, 9, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000) # set friction of gripper
        p.changeDynamics(self.robot_id, 10, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)

        # Set maximum joint velocity. Maximum joint velocity taken from:
        # https://s3-eu-central-1.amazonaws.com/franka-de-uploads/uploads/Datasheet-EN.pdf
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=0, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=1, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=2, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=3, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=4, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=5, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=6, maxJointVelocity=180 * (math.pi / 180))

        # Set DOF according to the fact that either gripper is supplied or not and create often used joint list
        self.dof = p.getNumJoints(self.robot_id) - 3
        self.joints = [0,1,2,3,4,5,6,9,10] # Set of non fixed joints(ignore the remaining ones)
        # Reset Robot
        self.counter = 0

        # storing the projection and view matrix
        self.height = 1024
        self.width = 1024
        self.sample_freq = sample_freq
        self.cameraDistance = 1.8  # How far away the camera is from the target
        self.cameraYaw = 250  # Camera rotation around the vertical axis in degrees
        self.cameraPitch = -40  # Camera vertical angle in degrees (-90 to 90)
        self.cameraTargetPosition = [+0.8, 0.4, 0.5]  # The XYZ position the camera is 
        self.max_steps = max_steps
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.cameraTargetPosition, distance=self.cameraDistance, yaw=self.cameraYaw, pitch=self.cameraPitch, roll=0, upAxisIndex=2)
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=90, aspect=float(self.width) / float(self.height), nearVal=0.12, farVal=100.0)
        self.prev_image = None
        self.new_image = None

        self.reset()

    def vip_image_transformation(self,image):

        transformed_image = Image.fromarray(image)
        transformed_image = transformed_image.resize((224, 224), Image.BICUBIC)
        transformed_image = transform(transformed_image)
        transformed_image = transformed_image.unsqueeze(0)
        transformed_image = transformed_image*255
        return transformed_image

    def reset(self, seed = 37):
        """"""

        for j in self.joints:
            p.resetJointState(self.robot_id, j, targetValue=0)

        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for _ in self.joints])
        
        self.counter = 0

        curr_pos, _ = self.get_position_and_velocity()
        if self.time_embedding == True:
            curr_pos += [0]
        p.stepSimulation()
        time.sleep(SAMPLING_RATE)

        info = {}
        self.prev_image = None
        self.new_image = p.getCameraImage(self.width, self.height, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return (curr_pos, info)

    def get_dof(self):
        """"""
        return self.dof

    def get_joint_info(self, j):
        """"""
        return p.getJointInfo(self.robot_id, j)

    def get_base_position_and_orientation(self):
        """"""
        return p.getBasePositionAndOrientation(self.robot_id)

    def get_position_and_velocity(self):
        """"""
        # self.joints is the index of all the joints whose information I want, the last joint is omitted here
        joint_states = p.getJointStates(self.robot_id, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        return joint_pos, joint_vel

    def calculate_inverse_kinematics(self, position, orientation=None):
        """"""
        if not orientation is None:
            return p.calculateInverseKinematics(self.robot_id, self.dof, position, orientation)
        else:
            return p.calculateInverseKinematics(self.robot_id, self.dof, position)

    def calculate_inverse_dynamics(self, pos, vel, desired_acc):
        """"""
        assert len(pos) == len(vel) and len(vel) == len(desired_acc)
        vector_length = len(pos)

        # If robot set up with gripper, set those positions, velocities and desired accelerations to 0
        if self.dof == 9 and vector_length != 9:
            pos = pos + [0., 0.]
            vel = vel + [0., 0.]
            desired_acc = desired_acc + [0., 0.]

        simulated_torque = list(p.calculateInverseDynamics(self.robot_id, pos, vel, desired_acc))

        # Remove unnecessary simulated torques for gripper if robot set up with gripper
        if self.dof == 9 and vector_length != 9:
            simulated_torque = simulated_torque[:7]
        return simulated_torque

    def set_target_positions(self, desired_pos):
        """"""

        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=desired_pos)

    def set_torques(self, desired_torque):
        """"""
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=desired_torque)

    def step(self, action=None):
        '''
        returns the next position, observation, reward, done, and info
        '''

        # global model
        self.counter += 1
        curr_pos, _ = self.get_position_and_velocity()
        next_pos = [0 for _ in range(len(curr_pos))]
        for i in range(len(action)):
            next_pos[i] = curr_pos[i] + action[i]
        self.set_target_positions(next_pos)
        p.stepSimulation()
        time.sleep(SAMPLING_RATE)

        reward = 0.0

        if (self.counter % self.sample_freq == 0):
            
            # The goal image transformations were done by the constructor and so it saves time by not doing it here again and again
            goal_image = self.goal_image

            with torch.no_grad():
                goal_embeddings = model(goal_image.to(device))
                goal_embeddings = goal_embeddings.cpu().numpy()

            self.prev_image = self.new_image
            self.new_image = p.getCameraImage(self.width, self.height, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            prev_rgb = self.prev_image
            prev_rgb = prev_rgb[2]
            prev_rgb_array = np.reshape(prev_rgb, (self.height, self.width, 4))
            prev_rgb_array = prev_rgb_array[:, :, :3]
            prev_rgb_array = self.vip_image_transformation(prev_rgb_array)
            with torch.no_grad():
                embeddings = model(prev_rgb_array.to(device))
                embeddings = embeddings.cpu().numpy()

            prev_distance = np.mean((goal_embeddings.squeeze()- embeddings.squeeze())**2)                    

            new_rgb = self.new_image
            new_rgb = new_rgb[2]          
            new_rgb_array = np.reshape(new_rgb, (self.height, self.width, 4))
            new_rgb_array = new_rgb_array[:, :, :3]
            new_rgb_array = self.vip_image_transformation(new_rgb_array)

            with torch.no_grad():
                embeddings = model(new_rgb_array.to(device))
                embeddings = embeddings.cpu().numpy()

            new_distance = np.mean((goal_embeddings.squeeze()- embeddings.squeeze())**2)              
            reward = ((prev_distance - new_distance).item())
        done = False
        if self.counter == self.max_steps:
            done = True

        new_pos, _ = self.get_position_and_velocity()
        if self.time_embedding == True:
            new_pos += [self.counter/self.max_steps]

        return new_pos, reward, done, done, {}