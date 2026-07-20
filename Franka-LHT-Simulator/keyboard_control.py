'''
Keyboard Teleportation File
This file is used to generate a trajectory of the robotic arm using keyboard controls.
This code only generates the csv file. To get the video, use the code on hydrogen, it takes this csv files simulates it and then generates video from it.
The csv file generated from this code is in the following format :

7 dimension joint space
1 dimension end effector gap
3 dimension task space
1 dimension time embedding
7 dimension action
'''

import pybullet as p
import time
import pybullet_data
import keyboard # type: ignore
import os
import csv
import math
import numpy as np
import pickle

object_location_dict = { # This is a mapping of object name to its urdf location
    "table"            : "./Objects/wood_table/urdf/wood_table.urdf",
    "rack"             : "./Objects/Rack/Rack.urdf",
    "bucket"           : "./Objects/bucket/bucketv02/bucketv02.urdf",
    "banana"           : "./Objects/banana/banana.urdf",
    "YcbGelatinBox"    : "./Objects/YcbGelatinBox/model.urdf",
    "carrot"           : "./Objects/carrot/carrot.urdf",
    "bucket_cap"       : "./Objects/bucket/bucket_v03/Bucket_cap_v02.urdf",
    "glass"            : "./Objects/glass/glass.urdf",
    "juice_bottle"     : "./Objects/juice_bottle/juice_bottle.urdf",
    "strawberry"       : "./Objects/strawberry/strawberry.urdf",
    "Tray"             : "./Objects/Tray/Tray.urdf",
    "bin"              : "./Objects/bin/bin.urdf",
    "syring"           : "./Objects/Syring/Syring.urdf",
    "syrup_bottle"     : "./Objects/syrup_bottle/syrup_bottle.urdf",
    "sunscreen"        : "./Objects/sunscreen/sunscreen.urdf",
    "box"              : "./Objects/open_cardboard box/cardboard.urdf",
    "wooden_crate"     : "./Objects/wooden_crate/wooden_crate.urdf",
    "bottle01"         : "./Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "bottle02"         : "./Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "bottle03"         : "./Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "bottle04"         : "./Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "YcbMustardBottle" : "./Objects/YcbMustardBottle/model.urdf",
    "kelloges"         : "./Objects/kelloges/kelloges.urdf",
    "dominosugar"      : "./Objects/dominosugar/dominosugar.urdf",
}

def set_object_intrinsic_properties(object_id, fixed_base , obj_name):
    if(not fixed_base):
        num_links = p.getNumJoints(object_id)
        for link_index in range(-1,num_links,1):
            p.changeDynamics(object_id, link_index, mass= (0.5/(num_links+1))) # This ensures the total mass of the objects is 0.1
            p.changeDynamics(object_id, link_index, lateralFriction=5, spinningFriction=5, rollingFriction=5) # This ensures small friction on the surface of the objects for avoiding unnecessary frictionless movements        
    else: # fixed objects
        if(obj_name == "table"):
            p.changeDynamics(object_id,-2,restitution=0.1,contactStiffness=500,contactDamping=300)

def domestic_assistive_tasks(p):
    object_dict={
        "table" : [[0,0,-0.42] , [-1,0,0,1], True],
        "rack"  : [[1.35,0.74,0.0], [0,0,-1,-1], True],
        "bucket" : [[0.65,1.2,0.74], [-1,0,0,-1], True],
        "banana" : [[0.20,0.7,0.75], [-1,0,1,1], False],
        "YcbGelatinBox" : [[0.35,0.9,0.80], [0,0,0,1], False],
        "carrot" : [[0.85,0.85,0.75], [-1,0,0,1], False],
        "bucket_cap" : [[0.3,1.2,0.77], [0,0,0,-1], False],
        "glass" : [[1.2,0.8,1.1], [0,0,0,1], False],
        "juice_bottle" : [[1.05,0.3,0.75], [0,0,0,1], False],
        "strawberry" : [[0.6,0.9,0.75], [-1,0,0,1], False],
    }

    for key in object_dict:
        obj = p.loadURDF(object_location_dict[key],object_dict[key][0], object_dict[key][1],useFixedBase = object_dict[key][2])
        set_object_intrinsic_properties(obj , object_dict[key][2] , key)
    return object_dict

def medical_waste_sorting(p):
    object_dict={
        "table" : [[0,0,-0.42] , [-1,0,0,1], True],
        "Tray"  : [[0.35,1.2,0.74], [0,0,0,-1], True],
        "bin" : [[1.02,0.9,0.74], [0,0,0,-1], True],
        "syring" : [[0.75,1.2,0.80], [-1,0,1,1], False],
        "syrup_bottle" : [[0.25,0.85,0.73], [0,0,0,1], False],
        "sunscreen" : [[0.5,1.0,0.89], [1,0,0,0], False]
    }

    for key in object_dict:
        obj = p.loadURDF(object_location_dict[key],object_dict[key][0], object_dict[key][1],useFixedBase = object_dict[key][2])
        set_object_intrinsic_properties(obj , object_dict[key][2] , key)

    return object_dict

def food_and_beverage_packing(p):
    object_dict={
        "table" : [[0,0,-0.42] , [-1,0,0,1], True],
        "box" : [[0.20,0.95,0.74], [0,0,0,-1] , True],
        "wooden_crate" : [[1.00,0.85,0.74], [0,0,1,-1], True],
        "bottle01" : [[0.96,0.85,0.75], [-1,0,0,1] , False],
        "bottle02" : [[0.96,0.62,0.75], [-1,0,0,1] , False],
        "bottle03" : [[0.96,1.07,0.73], [-1,0,0,1] , False],
        "bottle04" : [[0.80,1.07,0.73], [-1,0,0,1] , False],
        "YcbMustardBottle" : [[0.50,0.9,0.85], [0,0,-1,1], False],
        "kelloges" : [[0.45,1.20,0.93], [1,0,0,0], False],
        "dominosugar" : [[0.70,1.0,0.93] , [1,1,0,0], False]
    }

    for key in object_dict:
        obj = p.loadURDF(object_location_dict[key],object_dict[key][0], object_dict[key][1],useFixedBase = object_dict[key][2])
        set_object_intrinsic_properties(obj , object_dict[key][2] , key)

    return object_dict


if __name__ == "__main__": 
    scene = "scene3" # Describe the scene to be loaded in the simulator
    gaussian_noise=True
    std_dev =0.05  # Standard deviation of the Gaussian noise

    start_time = time.time()
    SAMPLING_RATE= 1.0/50.0 #1e-3
    physicsClient = p.connect(p.GUI)

    # p.addUserDebugPoints( [[1.05, 0.3 , 0.75]], [[1, 0, 0]], 15) # Add points for marker

    # Camera Parameters, these parameters dont matter since this is just csv generation
    cameraDistance = 1.8  # How far away the camera is from the target
    cameraYaw = 250  # Camera rotation around the vertical axis in degrees
    cameraPitch = -40  # Camera vertical angle in degrees (-90 to 90)
    cameraTargetPosition = [0.8, 0.4, 0.5]  # The XYZ position the camera is looking at

    p.setTimeStep(SAMPLING_RATE)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0)
    planeId = p.loadURDF("plane.urdf", [0, 0, -0.001])
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
    startOrientation = [0,0,0,1]

    if(scene == "scene1"):
        object_loc_dict = domestic_assistive_tasks(p)
    elif(scene == "scene2"):
        object_loc_dict = medical_waste_sorting(p)
    elif(scene == "scene3"):
        object_loc_dict = food_and_beverage_packing(p)

    franka_robot=p.loadURDF("./franka_emika_panda_pybullet-master/panda_robot/model_description/black_panda.urdf",[0.62,0.62,0.61], [0,0,1,1],useFixedBase =True , flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS )
    p.changeDynamics(franka_robot, 9, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)
    p.changeDynamics(franka_robot, 10, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=0, maxJointVelocity=150 * (math.pi / 180))
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=1, maxJointVelocity=150 * (math.pi / 180))
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=2, maxJointVelocity=150 * (math.pi / 180))
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=3, maxJointVelocity=150 * (math.pi / 180))
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=4, maxJointVelocity=180 * (math.pi / 180))
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=5, maxJointVelocity=180 * (math.pi / 180))
    p.changeDynamics(bodyUniqueId=franka_robot, linkIndex=6, maxJointVelocity=180 * (math.pi / 180))

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, "q","w","e","r","t","y","u","i","o"]

    print("Total number of joints = ", p.getNumJoints(franka_robot))
    for i in range(p.getNumJoints(franka_robot)):
        joint_info = p.getJointInfo(franka_robot, i)
        # Check if the joint type is p.JOINT_FIXED
        is_fixed_joint = joint_info[2] == p.JOINT_FIXED

        if is_fixed_joint:
            print(f"Joint {i} is a fixed joint.")
        else:
            print(f"Joint {i} is not a fixed joint.")

    dict = {0 : [7,+1.0] ,
            1 : [0,+1.0],
            2 : [1,+1.0],
            3 : [2,1.0],
            4 : [3,1.0],
            5 : [4,1.0],
            6 : [5,1.0],
            7 : [6,1.0],
            8 : [9,1.0],
            'q' : [0,-1.0],
            'w' : [1,-1.0],
            'e' : [2,-1.0],
            'r' : [3,-1.0],
            't' : [4,-1.0],
            'y' : [5,-1.0],
            'u' : [6,-1.0],
            'i' : [9,-1.0],
            'o' : [9,-.05],
            }

    done=False
    non_fixed_joints = [0,1,2,3,4,5,6,9,10]

    reset_dict_original = {0:0.0 , 1:-0.7853981633974483 ,2:0.0 , 3: -2.356194490192345, 4:0.0 , 5: 1.5707963267948966, 6:0.7853981633974483 , 9:0.0 , 10:0.0  } # This is used for coming back to home
    reset_dict = {0:0.0 , 1:-0.7853981633974483 ,2:0.0 , 3: -2.356194490192345, 4:0.0 , 5: 1.5707963267948966, 6:0.7853981633974483 , 9:0.0 , 10:0.0  }
    if(gaussian_noise):
        mean = 0  # Mean of the Gaussian noise
        reset_dict = {key: value + np.random.normal(mean, abs(value) * std_dev) for key, value in reset_dict.items()}
        print("reset_dict = ", reset_dict)

    for j in reset_dict:
        p.resetJointState(franka_robot, j, targetValue=reset_dict[j])

    all_states = []
    all_actions = []
    action_multiplier= 0.1 

    dict_1 = {0:1,1:2,2:3,3:4,4:5,5:6,6:7,9:8} # joint -> button which makes the joint increase
    dict_2 = {0:'q',1:'w',2:'e',3:'r',4:'t',5:'y',6:'u',9:'i'}

    while True:
        current_joint_state = [state[0] for state in p.getJointStates(franka_robot, range(12))]
        # These are recommendations to get back to the reset position at the end
        print("---------------To Reach Reset Position----------------")
        for i in [0,1,2,3,4,5,6,9]:
            if(current_joint_state[i] - reset_dict_original[i] >0.01):
                print(f"Press key {dict_2[i]}")
            elif (current_joint_state[i] - reset_dict_original[i] < -0.01):
                print(f"Press key {dict_1[i]}")

        action = [0 for _ in range(12)]
        for joint_index in joint_indices:
            # Adjust the joint positions based on key events
            if keyboard.is_pressed(f"{joint_index}"):
                if(joint_index==0):
                    done=True
                current_position = current_joint_state[dict[joint_index][0]]
                target_position = current_position +dict[joint_index][1]*action_multiplier  # Adjust the increment as needed
                action[dict[joint_index][0]] = dict[joint_index][1]*action_multiplier

        non_fixed_current_joint_state = [current_joint_state[i] for i in non_fixed_joints]
        non_fixed_action = [action[i] for i in non_fixed_joints]
        non_fixed_action[-1] = non_fixed_action[-2] # Action on right end effector same as action on left end effector
        non_fixed_target = [0 for _ in range(len(non_fixed_current_joint_state))]

        for ind in range(len(non_fixed_current_joint_state)):
            non_fixed_target[ind] = non_fixed_current_joint_state[ind] + non_fixed_action[ind]

        p.setJointMotorControlArray(bodyUniqueId=franka_robot,
                                        jointIndices=non_fixed_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=non_fixed_target, targetVelocities = [0 for _ in non_fixed_action], forces=[1000. for _ in non_fixed_joints])

        non_fixed_current_joint_state.pop() # Remove the state of right joint, it must be same as left joint
        non_fixed_action.pop() # Remove the state of right joint, it must be same as left joint
        end_effector_pose = p.getLinkState(franka_robot, 11, computeForwardKinematics=True)
        end_effector_position = end_effector_pose[0]

        non_fixed_current_joint_state.extend(end_effector_position)
        # 0 state action will also be appended here
        all_states.append(non_fixed_current_joint_state)
        all_actions.append(non_fixed_action)

        p.stepSimulation()
        time.sleep(0.0)  # Adjust sleep time based on your desired control rate , 0.01 initially
        if(done):
            break

    base_directory = 'Trajectory'
    existing_directories = [int(d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    largest_number = max(existing_directories) if existing_directories else 0
    # Create a new directory with the next number
    new_directory = os.path.join(base_directory, str(largest_number + 1))
    os.makedirs(new_directory)

    # Create and save CSV file in the new directory
    csv_file_path = os.path.join(new_directory, 'data.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for index, (state, action) in enumerate(zip(all_states, all_actions), start=0):
            csv_writer.writerow( state + [index]+ action)
    
    pkl_file_path = os.path.join(new_directory, 'location.pkl')
    with open(pkl_file_path, "wb") as f:
        pickle.dump({"object_loc_dict": object_loc_dict, "reset_dict": reset_dict , "length" : len(all_states)}, f)

    p.disconnect()