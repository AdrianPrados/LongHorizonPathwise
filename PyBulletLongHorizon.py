import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
import math

import pybullet as p
import pybullet_data

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates

# Tu librería local
from ProGP import ProGpMp

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'cyan']

# ============================================================================
# PARTE 1: FUNCIONES DE ENTORNO DE PYBULLET
# ============================================================================

object_location_dict = {
    "table"            : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/wood_table/urdf/wood_table.urdf",
    "rack"             : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/Rack/Rack.urdf",
    "bucket"           : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/bucket/bucketv02/bucketv02.urdf",
    "banana"           : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/banana/banana.urdf",
    "YcbGelatinBox"    : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/YcbGelatinBox/model.urdf",
    "carrot"           : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/carrot/carrot.urdf",
    "bucket_cap"       : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/bucket/bucket_v03/Bucket_cap_v02.urdf",
    "glass"            : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/glass/glass.urdf",
    "juice_bottle"     : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/juice_bottle/juice_bottle.urdf",
    "strawberry"       : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/strawberry/strawberry.urdf",
    "Tray"             : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/Tray/Tray.urdf",
    "bin"              : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/bin/bin.urdf",
    "syring"           : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/Syring/Syring.urdf",
    "syrup_bottle"     : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/syrup_bottle/syrup_bottle.urdf",
    "sunscreen"        : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/sunscreen/sunscreen.urdf",
    "box"              : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/open_cardboard box/cardboard.urdf",
    "wooden_crate"     : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/wooden_crate/wooden_crate.urdf",
    "bottle01"         : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "bottle02"         : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "bottle03"         : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "bottle04"         : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/plastic_water_bottle/urdf/plastic_water_bottle.urdf",
    "YcbMustardBottle" : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/YcbMustardBottle/model.urdf",
    "kelloges"         : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/kelloges/kelloges.urdf",
    "dominosugar"      : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/dominosugar/dominosugar.urdf",
}

def set_object_intrinsic_properties(object_id, fixed_base, obj_name):
    if not fixed_base:
        num_links = p.getNumJoints(object_id)
        for link_index in range(-1, num_links, 1):
            p.changeDynamics(object_id, link_index, mass=(0.1/(num_links+1))) 
            p.changeDynamics(object_id, link_index, lateralFriction=50, spinningFriction=50, rollingFriction=50, frictionAnchor=1)
    else:
        if obj_name == "table":
            p.changeDynamics(object_id, -2, restitution=0.1, contactStiffness=500, contactDamping=300)

def calibrate_object_to_ground(object_id,table_id):

    _, table_aabb_max = p.getAABB(table_id)
    table_top_z = table_aabb_max[2] + 0.035
    obj_aabb_min, _ = p.getAABB(object_id)
    obj_base_z = obj_aabb_min[2]
    offset = obj_base_z - table_top_z
    pos, ori = p.getBasePositionAndOrientation(object_id)
    new_pos = [pos[0], pos[1], pos[2] - offset]
    p.resetBasePositionAndOrientation(object_id, new_pos, ori)

# Scenes
def domestic_assistive_tasks():
    object_dict = {
        "table": [[0,0,-0.42], [-1,0,0,1], True],
        "rack": [[1.35,0.74,0.0], [0,0,-1,-1], True],
        "bucket": [[0.65,1.2,0.74], [-1,0,0,-1], True],
        "banana": [[0.20,0.7,0.77], [-1,0,1,1], False],
        "YcbGelatinBox": [[0.35,0.9,0.80], [0,0,0,1], False],
        "carrot": [[0.85,0.85,0.75], [-1,0,0,1], False],
        "bucket_cap": [[0.3,1.2,0.77], [0,0,0,-1], False],
        "glass": [[1.2,0.8,1.1], [0,0,0,1], False],
        "juice_bottle": [[1.05,0.3,0.75], [0,0,0,1], False],
        "strawberry": [[0.6,0.9,0.75], [-1,0,0,1], False],
    }
    loaded_objects = {}
    for key in object_dict:
        obj = p.loadURDF(object_location_dict[key], object_dict[key][0], object_dict[key][1], useFixedBase=object_dict[key][2])
        set_object_intrinsic_properties(obj, object_dict[key][2], key)
        loaded_objects[key] = obj
    """ table_id = loaded_objects['table']
    for key, obj_id in loaded_objects.items():
        if not object_dict[key][2]:
            calibrate_object_to_ground(obj_id, table_id) """
    return loaded_objects

def medical_waste_sorting():
    object_dict={
        "table" : [[0,0,-0.42] , [-1,0,0,1], True],
        "Tray"  : [[0.35,1.2,0.74], [0,0,0,-1], True],
        "bin" : [[1.02,0.9,0.74], [0,0,0,-1], True],
        "syring" : [[0.75,1.2,0.80], [-1,0,1,1], False],
        "syrup_bottle" : [[0.25,0.85,0.73], [0,0,0,1], False],
        "sunscreen" : [[0.5,1.0,0.89], [1,0,0,0], False]
    }
    loaded_objects = {}
    for key in object_dict:
        obj = p.loadURDF(object_location_dict[key],object_dict[key][0], object_dict[key][1],useFixedBase = object_dict[key][2])
        set_object_intrinsic_properties(obj , object_dict[key][2] , key)
        loaded_objects[key] = obj
    """ table_id = loaded_objects['table']
    for key, obj_id in loaded_objects.items():
        if not object_dict[key][2]:
            calibrate_object_to_ground(obj_id, table_id) """
    return loaded_objects

def food_and_beverage_packing():
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
    loaded_objects = {}
    for key in object_dict:
        obj = p.loadURDF(object_location_dict[key],object_dict[key][0], object_dict[key][1],useFixedBase = object_dict[key][2])
        set_object_intrinsic_properties(obj , object_dict[key][2] , key)
        loaded_objects[key] = obj
    """ table_id = loaded_objects['table']
    for key, obj_id in loaded_objects.items():
        if not object_dict[key][2]:
            calibrate_object_to_ground(obj_id, table_id) """
    return loaded_objects

# ============================================================================
# PARTE 2: GRÁFICO INTERACTIVO MATPLOTLIB Y CONTROLADOR PYBULLET
# ============================================================================

class PyBulletGPController:
    def __init__(self, segments_info, all_prior, all_corrected, robot_id, ee_idx, loaded_objects):
        self.segments_info = segments_info
        self.all_prior = all_prior
        self.all_corrected = all_corrected
        self.lines = []
        self.loaded_objects = loaded_objects
        
        self.robot_id = robot_id
        self.ee_idx = ee_idx
        self.current_step = 0
        self.full_trajectory = None

        self.initial_object_states = {}
        for obj_name, obj_id in self.loaded_objects.items():
            pos, ori = p.getBasePositionAndOrientation(obj_id)
            self.initial_object_states[obj_name] = {'pos': pos, 'ori': ori}
            
        self.robot_initial_joints = {0:0.0, 1:-0.7853981633974483, 2:0.0, 3:-2.356194490192345, 4:0.0, 5:1.5707963267948966, 6:0.7853981633974483, 9:0.04, 10:0.04}

        self.gripper_width = 0.04           
        self.grasp_state = "IDLE"           
        self.pause_counter = 0
        self.grasped_objects = set()        
        self.dropped_objects = set() 
         
        self.current_target_obj = None
        self.grasp_constraint = None 

        self.points = []
        self.point_mapping = [] 
        for seg_idx, info in enumerate(segments_info):
            for v_idx, pt in enumerate(info['vias_cond']):
                self.points.append(pt)
                self.point_mapping.append((seg_idx, v_idx))
                
        self.points = np.array(self.points, float)
        self.last_points = self.points.copy()
        
        self.label_offsets = {}
        for i in range(len(self.points)):
            seg_idx, v_idx = self.point_mapping[i]
            label = self.segments_info[seg_idx]['obj_names'].get(v_idx, "")
            if label in self.loaded_objects:
                base_pos, _ = p.getBasePositionAndOrientation(self.loaded_objects[label])
                self.label_offsets[label] = self.points[i] - np.array(base_pos)

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)
        self.ax.set_title('Visualizador Pasivo de Trayectoria\n(Mueve los sliders en PyBullet)', fontsize=14)

        for i in range(len(segments_info)):
            self.ax.plot(all_prior[i][:, 0], all_prior[i][:, 1], all_prior[i][:, 2], '--', linewidth=2, c='grey', alpha=0.4)
            line, = self.ax.plot(all_corrected[i][:, 0], all_corrected[i][:, 1], all_corrected[i][:, 2], '-', linewidth=3, c=colors[i % len(colors)])
            self.lines.append(line)

        point_colors = []
        for i in range(len(self.points)):
            seg_idx, v_idx = self.point_mapping[i]
            label = self.segments_info[seg_idx]['obj_names'].get(v_idx, "")
            
            if 'drop_point' in label:
                point_colors.append('red')
            elif label in self.loaded_objects:
                point_colors.append('orange')
            else:
                point_colors.append('blue')

        self.scatter = self.ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], s=150, c=point_colors, marker='D')

        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='orange', marker='D', linestyle='None', markersize=10, label='Via-point (Objeto)'))
        handles.append(mlines.Line2D([], [], color='red', marker='D', linestyle='None', markersize=10, label='Virtual Drop Point'))
        handles.append(mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Nodo Base'))
        self.ax.legend(handles=handles, fontsize=10, loc='best')

        self._build_full_trajectory()
        
        self.slider_groups = []
        self.last_slider_reads = [] 
        visited_indices = set()
        
        for i in range(len(self.points)):
            if i in visited_indices: continue
            shared_idxs = [j for j in range(len(self.points)) if np.linalg.norm(self.points[j] - self.points[i]) < 1e-4]
            visited_indices.update(shared_idxs)
            
            seg_idx, v_idx = self.point_mapping[i]
            label = self.segments_info[seg_idx]['obj_names'].get(v_idx, f"Nodo_{i}")
            init_pos = self.points[i]
            
            sx = p.addUserDebugParameter(f"{label} [X]", init_pos[0]-0.6, init_pos[0]+0.6, init_pos[0])
            sy = p.addUserDebugParameter(f"{label} [Y]", init_pos[1]-0.6, init_pos[1]+0.6, init_pos[1])
            sz = p.addUserDebugParameter(f"{label} [Z]", 0.1, 1.5, init_pos[2])
            
            self.slider_groups.append({'idxs': shared_idxs, 'sliders': (sx, sy, sz), 'label': label})
            self.last_slider_reads.append(init_pos.copy())

        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self.step_simulation)
        self.timer.start()

    def _build_full_trajectory(self):
        full = []
        self.action_schedule = {}  
        current_offset = 0
        
        for i, info in enumerate(self.segments_info):
            mean_corr = info.get('mean_corrected', self.all_corrected[i])
            full.append(mean_corr)
            
            test_x = info['test_x']
            via_t_cond = info['via_t_cond']
            obj_names = info['obj_names']
            
            for v_idx, t in enumerate(via_t_cond):
                if v_idx in obj_names:
                    label = obj_names[v_idx]
                    local_idx = int(np.argmin(np.abs(test_x.flatten() - t)))
                    global_idx = current_offset + local_idx
                    
                    if 'drop_point' in label:
                        target_name = label.replace('drop_point_', '')
                        self.action_schedule[global_idx] = {"action": "DROP", "target": target_name}
                    else:
                        self.action_schedule[global_idx] = {"action": "GRASP", "target": label}
                        
            current_offset += len(mean_corr)
            
        self.full_trajectory = np.vstack(full)
        print(f"[Planificador] Acciones programadas en los steps: {self.action_schedule}")

    def step_simulation(self):
        changed = False
        segments_to_update = set()
        
        for group_idx, group in enumerate(self.slider_groups):
            sx, sy, sz = group['sliders']
            new_pos = np.array([p.readUserDebugParameter(sx), p.readUserDebugParameter(sy), p.readUserDebugParameter(sz)])
            label = group['label']
            
            if np.linalg.norm(new_pos - self.last_slider_reads[group_idx]) > 1e-3:
                changed = True
                self.last_slider_reads[group_idx] = new_pos.copy()
                
                if label in self.loaded_objects and label not in self.grasped_objects:
                    obj_id = self.loaded_objects[label]
                    _, ori = p.getBasePositionAndOrientation(obj_id)
                    offset = self.label_offsets.get(label, np.zeros(3))
                    p.resetBasePositionAndOrientation(obj_id, new_pos - offset, ori)
                    p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])

                for idx in group['idxs']:
                    self.points[idx] = new_pos
                    seg_idx, v_idx = self.point_mapping[idx]
                    self.segments_info[seg_idx]['vias_cond'][v_idx] = new_pos
                    segments_to_update.add(seg_idx)
        
        if changed:
            for seg_idx in segments_to_update:
                self.recompute_segment(seg_idx)
            self._build_full_trajectory()
            self.scatter._offsets3d = (self.points[:, 0], self.points[:, 1], self.points[:, 2])
            self.fig.canvas.draw_idle()
            
        if self.full_trajectory is not None and len(self.full_trajectory) > 0:
            target_pos = self.full_trajectory[self.current_step]
            target_ori = p.getQuaternionFromEuler([math.pi, 0, 0])
            
            if self.grasp_state == "IDLE":
                if self.current_step in self.action_schedule:
                    action_info = self.action_schedule[self.current_step]
                    
                    if action_info["action"] == "GRASP" and self.grasp_constraint is None:
                        obj_name = action_info["target"]
                        if obj_name in self.loaded_objects and obj_name not in self.grasped_objects:
                            self.grasp_state = "WAITING"
                            self.pause_counter = 25 
                            self.current_target_obj = obj_name
                            
                    elif action_info["action"] == "DROP" and self.grasp_constraint is not None:
                        target_name = action_info["target"]
                        self.grasp_state = "RELEASING"
                        self.pause_counter = 20
                            
            elif self.grasp_state == "WAITING":
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    obj_id = self.loaded_objects[self.current_target_obj]
                    for finger_link in [9, 10, 11]:
                        p.setCollisionFilterPair(self.robot_id, obj_id, finger_link, -1, enableCollision=0)

                    self.gripper_width = 0.0  
                    self.grasp_state = "CLOSING"
                    self.pause_counter = 20 
                    
            elif self.grasp_state == "CLOSING":
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    obj_id = self.loaded_objects[self.current_target_obj]
                    
                    obj_pos, obj_ori = p.getBasePositionAndOrientation(obj_id)
                    ee_state = p.getLinkState(self.robot_id, self.ee_idx)
                    ee_pos, ee_ori = ee_state[4], ee_state[5] 
                    
                    inv_ee_pos, inv_ee_ori = p.invertTransform(ee_pos, ee_ori)
                    snap_rel_pos, rel_ori = p.multiplyTransforms(inv_ee_pos, inv_ee_ori, obj_pos, obj_ori)
                    
                    self.grasp_constraint = p.createConstraint(
                        parentBodyUniqueId=self.robot_id,
                        parentLinkIndex=self.ee_idx,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=snap_rel_pos,
                        childFramePosition=[0, 0, 0],
                        parentFrameOrientation=rel_ori
                    )
                    
                    p.changeConstraint(self.grasp_constraint, maxForce=5000)
                    self.grasped_objects.add(self.current_target_obj)
                    self.grasp_state = "IDLE"
                    time.sleep(0.5) 

            elif self.grasp_state == "RELEASING":
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    self.gripper_width = 0.04  
                    if self.grasp_constraint is not None:
                        p.removeConstraint(self.grasp_constraint)
                        self.grasp_constraint = None
                        
                    if self.current_target_obj:
                        obj_id = self.loaded_objects[self.current_target_obj]
                        for finger_link in [9, 10, 11]:
                            p.setCollisionFilterPair(self.robot_id, obj_id, finger_link, -1, enableCollision=1)
                            
                        self.dropped_objects.add(self.current_target_obj)
                        
                    self.grasped_objects.clear()
                    self.current_target_obj = None
                    self.grasp_state = "IDLE"
            
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.ee_idx, targetPosition=target_pos, targetOrientation=target_ori, maxNumIterations=1000
            )
            
            all_joints = [0, 1, 2, 3, 4, 5, 6, 9, 10]
            target_positions = list(joint_poses[:7]) + [self.gripper_width, self.gripper_width]
            forces = [300.] * 7 + [5., 5.] 
            
            for _ in range(15):
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot_id, jointIndices=all_joints,
                    controlMode=p.POSITION_CONTROL, targetPositions=target_positions, forces=forces
                )
                
                for group_idx, group in enumerate(self.slider_groups):
                    label = group['label']
                    
                    if label in self.loaded_objects:
                        if label not in self.grasped_objects and label not in self.dropped_objects:
                            obj_id = self.loaded_objects[label]
                            _, ori = p.getBasePositionAndOrientation(obj_id)
                            offset = self.label_offsets.get(label, np.zeros(3))
                            p.resetBasePositionAndOrientation(obj_id, self.last_slider_reads[group_idx] - offset, ori)
                            p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])
                
                p.stepSimulation()
            
            if self.grasp_state == "IDLE":
                self.current_step += 1
                if self.current_step >= len(self.full_trajectory):
                    self.current_step = 0
                    self.gripper_width = 0.04
                    if self.grasp_constraint is not None:
                        p.removeConstraint(self.grasp_constraint)
                        self.grasp_constraint = None
                        
                    self.grasped_objects.clear()
                    self.dropped_objects.clear()
                    self.current_target_obj = None
                    
                    for obj_name, obj_id in self.loaded_objects.items():
                        slider_pos = None
                        for group_idx, group in enumerate(self.slider_groups):
                            if group['label'] == obj_name:
                                slider_pos = self.last_slider_reads[group_idx]
                                break
                        
                        init_state = self.initial_object_states[obj_name]
                        if slider_pos is not None:
                            offset = self.label_offsets.get(obj_name, np.zeros(3))
                            p.resetBasePositionAndOrientation(obj_id, slider_pos - offset, init_state['ori'])
                        else:
                            p.resetBasePositionAndOrientation(obj_id, init_state['pos'], init_state['ori'])
                        p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])
                        
                    for j, target_val in self.robot_initial_joints.items():
                        p.resetJointState(self.robot_id, j, targetValue=target_val)

    def recompute_segment(self, seg_idx):
        info = self.segments_info[seg_idx]
        kernel = info['kernel']
        F_prior_full = info['F_prior_full']
        via_t_cond = info['via_t_cond']
        vias_cond = info['vias_cond']
        mean_gmp_arr = info['mean_gmp_arr']
        test_x = info['test_x']

        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_cond, dtype=default_float()), axis=0)

        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        F_obs = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0)

        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
        X_test = tf.convert_to_tensor(test_x, dtype=default_float())
        dF_full = dF_fn(X_test)
        F_cond = F_prior_full + dF_full

        mean_corrected = np.squeeze(F_cond.numpy())
        self.lines[seg_idx].set_data_3d(mean_corrected[:,0], mean_corrected[:,1], mean_corrected[:,2])
        self.segments_info[seg_idx]['mean_corrected'] = mean_corrected
        
    def show(self):
        plt.show()

# ============================================================================
# PARTE 3: LÓGICA DE PROCESOS GAUSSIANOS Y MAIN
# ============================================================================

def load_segment_data(base_dir, scene, task, demo, segment_idx, gap=10):
    folder_path = os.path.join(base_dir, f"Scene_{scene}_Task_{task}")
    file_name = f"demo_{demo}_segmento_{segment_idx}.txt"
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path): raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    pos_data_full = np.loadtxt(file_path)
    pos_data = pos_data_full[::gap, :]
    dt = 0.01 
    num_points = pos_data.shape[0]
    return np.arange(0, num_points * dt, dt).reshape(-1, 1), pos_data, dt

def draw_debug_target(pos, label, is_grasp=True):
    color = [0, 1, 0] if is_grasp else [1, 0, 0] 
    length = 0.04
    p.addUserDebugLine([pos[0]-length, pos[1], pos[2]], [pos[0]+length, pos[1], pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1]-length, pos[2]], [pos[0], pos[1]+length, pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1], pos[2]-length], [pos[0], pos[1], pos[2]+length], color, 3)
    p.addUserDebugText(label, [pos[0], pos[1], pos[2]+0.05], textColorRGB=color, textSize=1.2)

def chain_gp_pathwise_segments(scene=1, task=1, demo=1, manual_objects=None, robot_id=None, ee_idx=11):
    
    pkl_path = f'/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Trajectory/{scene}.{task}.{demo}/location.pkl'
    
    if(scene == 1):
        loaded_objects = domestic_assistive_tasks()
    elif(scene == 2):
        loaded_objects = medical_waste_sorting()
    elif(scene == 3):
        loaded_objects = food_and_beverage_packing()
    else:
        print(f"[Error] Unknow scene: {scene}. loading scene 1.")
        loaded_objects = domestic_assistive_tasks()
    
    base_dir = '/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Segmentation-and-Grouping-Model/SegmentationProb/scripts/LHT_SegmentsFolder' 
    print(f"--- Procesando Cadena S{scene} | T{task} | Demo {demo} ---")

    if not os.path.exists(pkl_path): object_loc_dict = {}
    else:
        with open(pkl_path, 'rb') as f: object_loc_dict = pickle.load(f).get('object_loc_dict', {})

    previous_end_pos = None
    all_prior, all_corrected, segments_info = [], [], []

    # =========================================================================
    # FASE 0: OBTENCIÓN DINÁMICA DE COORDENADAS (APRENDIZAJE DE MÍNIMO GLOBAL)
    # =========================================================================
    target_positions = {}
    global_min_dist = {}
    
    if manual_objects:
        # Pre-calculamos las posiciones físicas reales de los objetivos
        for obj_name, action_type in manual_objects.items():
            if obj_name in loaded_objects:
                obj_id = loaded_objects[obj_name]
                aabb_min, aabb_max = p.getAABB(obj_id)
                center_x = (aabb_max[0] + aabb_min[0]) / 2.0
                center_y = (aabb_max[1] + aabb_min[1]) / 2.0
                z_top = aabb_max[2]
                
                target_positions[obj_name] = []
                
                # Drop point esception for wooden_crate
                if obj_name == 'wooden_crate':
                    dx = (aabb_max[0] - aabb_min[0]) / 4.0
                    dy = (aabb_max[1] - aabb_min[1]) / 4.0
                    puntos_base = [
                        [center_x + dx, center_y + dy, z_top],
                        [center_x + dx, center_y - dy, z_top],
                        [center_x - dx, center_y + dy, z_top],
                        [center_x - dx, center_y - dy, z_top]
                    ]
                else:
                    puntos_base = [[center_x, center_y, z_top]]
                
                for pt in puntos_base:
                    t_pos = np.array(pt)
                    if action_type == 'drop': t_pos[2] += 0.25
                    elif action_type == 'grasp': t_pos[2] -= 0.02
                    target_positions[obj_name].append(t_pos)
                    
                global_min_dist[obj_name] = float('inf')
        
        seg_check = 1
        while True:
            try:
                X_c, Y_c, _ = load_segment_data(base_dir, scene, task, demo, seg_check)
                Y_c = Y_c[:min(X_c.shape[0], Y_c.shape[0])]
                
                for obj_name, t_pos_list in target_positions.items():
                    for t_pos in t_pos_list:
                        min_d = np.min(np.linalg.norm(Y_c - t_pos, axis=1))
                        if min_d < global_min_dist[obj_name]:
                            global_min_dist[obj_name] = min_d
            except FileNotFoundError:
                break
            seg_check += 1

    # =========================================================================
    # FASE 1: ASIGNACIÓN SEGMENTO A SEGMENTO BASADA EN LO APRENDIDO
    # =========================================================================
    segment_matches = {}
    if manual_objects:
        seg_check = 1
        while True:
            try:
                X_c, Y_c, _ = load_segment_data(base_dir, scene, task, demo, seg_check)
                min_len = min(X_c.shape[0], Y_c.shape[0])
                X_c, Y_c = X_c[:min_len], Y_c[:min_len]
            except FileNotFoundError:
                break 
            
            segment_matches[seg_check] = []
            for obj_name, t_pos_list in target_positions.items():
                
                best_dist = float('inf')
                best_idx = -1
                best_t_pos = None
                
                # Buscamos de nuevo a cuál de los sub-puntos (si hay varios) se acercó en ESTE segmento
                for t_pos in t_pos_list:
                    distances = np.linalg.norm(Y_c - t_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    min_dist_in_segment = distances[closest_idx]
                    
                    if min_dist_in_segment < best_dist:
                        best_dist = min_dist_in_segment
                        best_idx = closest_idx
                        best_t_pos = t_pos
                
                # LA CLAVE: Si en este segmento el robot pasa a una distancia muy similar 
                # a la mejor distancia absoluta (+ 8 cm de margen por variabilidad), es una iteración real.
                # Esto soporta MÚLTIPLES visitas a la misma caja en distintos segmentos.
                if best_dist <= global_min_dist[obj_name] + 0.08 and best_dist < 0.25:
                    action_type = manual_objects[obj_name]
                    segment_matches[seg_check].append({
                        'obj_name': obj_name,
                        'time': X_c[best_idx, 0],
                        'target_pos': best_t_pos,
                        'action_type': action_type
                    })
                    print(f"   [FASE 1] -> Detectado '{obj_name}' en Seg {seg_check} (dist: {best_dist:.3f}m)")
            
            seg_check += 1

    # =========================================================================
    # SEGUNDA FASE: CONSTRUCCIÓN DE PROCESOS GAUSSIANOS CON ASIGNACIÓN DIRECTA
    # =========================================================================
    segment_idx = 1
    while True:
        try:
            X, Y, dt = load_segment_data(base_dir, scene, task, demo, segment_idx)
            min_len = min(X.shape[0], Y.shape[0])
            X, Y = X[:min_len], Y[:min_len]
            size = X.shape[0]
        except FileNotFoundError:
            break
            
        print(f">> Aprendiendo Segmento {segment_idx}...")
        
        target_t = X[-1, 0]
        orig_start, orig_end = Y[0, :], Y[-1, :]
        via_t = np.array([X[0, 0], target_t]).reshape(-1, 1)
        vias_orig = np.array([orig_start, orig_end])
        
        curr_start = orig_start if previous_end_pos is None else previous_end_pos
        curr_end = orig_end
        previous_end_pos = curr_end
        
        gp_mp = ProGpMp(X, Y, via_t, vias_orig, dim=3, demos=1, size=size, observation_noise=0.01)
        gp_mp.BlendedGpMp(gp_mp.ProGP)
        
        test_x = np.linspace(X[0, 0], X[-1, 0], size * 4).reshape(-1, 1)
        mean_gmp, _ = gp_mp.predict_BlendedPos(test_x)

        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        base_kernels = [cls(lengthscales=0.5) for cls in BaseClasses] 
        kernel = kernels.SeparateIndependent(base_kernels)

        mean_gmp_arr = np.vstack(mean_gmp) 
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

        points_dict = {
            via_t[0, 0]: (curr_start, "Start"),
            target_t: (curr_end, "End")
        }

        if manual_objects and segment_idx in segment_matches:
            for match in segment_matches[segment_idx]:
                obj_name = match['obj_name']
                closest_time = match['time']
                target_pos = match['target_pos']
                action_type = match['action_type']
                
                virtual_name = obj_name
                if action_type == 'drop': 
                    virtual_name = f'drop_point_{obj_name}'
                    draw_debug_target(target_pos, f"DROP {obj_name}", is_grasp=False)
                elif action_type == 'grasp': 
                    draw_debug_target(target_pos, f"GRASP {obj_name}", is_grasp=True)
                
                if abs(closest_time - via_t[0, 0]) < 0.05:
                    points_dict[via_t[0, 0]] = (target_pos, virtual_name)
                    print(f"   [VIA-POINT ASIGNADO] -> '{virtual_name}' fusionado en Start t={via_t[0, 0]:.4f}s")
                elif abs(closest_time - target_t) < 0.05:
                    points_dict[target_t] = (target_pos, virtual_name)
                    print(f"   [VIA-POINT ASIGNADO] -> '{virtual_name}' fusionado en End t={target_t:.4f}s")
                else:
                    while closest_time in points_dict:
                        closest_time += 1e-4  
                    points_dict[closest_time] = (target_pos, virtual_name)
                    print(f"   [VIA-POINT ASIGNADO] -> '{virtual_name}' fijado internamente en t={closest_time:.4f}s")

        points_info_list = [(t, val[0], val[1]) for t, val in points_dict.items()]
        points_info_list.sort(key=lambda x: x[0])

        via_t_cond = np.array([p[0] for p in points_info_list])
        vias_cond = np.array([p[1] for p in points_info_list])
        obj_names_map = {idx: p[2] for idx, p in enumerate(points_info_list) if p[2] not in ["Start", "End"]}

        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_cond, dtype=default_float()), axis=0)
        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        F_obs = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 

        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
        X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 
        F_cond = F_prior_full + dF_fn(X_test)

        mean_corrected = np.squeeze(F_cond.numpy())
        mean_prior = np.squeeze(mean_gmp_arr.T)
        
        all_prior.append(mean_prior)
        all_corrected.append(mean_corrected)

        segments_info.append({
            'kernel': kernel, 'F_prior_full': F_prior_full, 'via_t_cond': via_t_cond,
            'vias_cond': vias_cond, 'mean_gmp_arr': mean_gmp_arr, 'test_x': test_x,
            'obj_names': obj_names_map, 'mean_corrected': mean_corrected 
        })
        segment_idx += 1

    if segment_idx - 1 > 0:
        print("\n>> Abriendo Visor Interactivo y Controles en PyBullet...")
        plotter = PyBulletGPController(segments_info, all_prior, all_corrected, robot_id, ee_idx, loaded_objects)
        plotter.show()

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    planeId = p.loadURDF("plane.urdf", [0, 0, -0.001])
    p.resetDebugVisualizerCamera(1.8, 250, -40, [0.8, 0.4, 0.5])

    franka_robot = p.loadURDF(
        "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/franka_emika_panda_pybullet-master/panda_robot/model_description/black_panda.urdf",
        [0.62, 0.62, 0.61], [0,0,1,1], 
        useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    )
    
    p.changeDynamics(franka_robot, 9, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000, frictionAnchor=1)
    p.changeDynamics(franka_robot, 10, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000, frictionAnchor=1)
    p.changeDynamics(franka_robot, 11, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000, frictionAnchor=1)
    
    reset_dict_original = {0:0.0, 1:-0.7853981633974483, 2:0.0, 3:-2.356194490192345, 4:0.0, 5:1.5707963267948966, 6:0.7853981633974483, 9:0.04, 10:0.04}
    for j in reset_dict_original:
        p.resetJointState(franka_robot, j, targetValue=reset_dict_original[j])

    # Variables de control unificadas
    CURRENT_SCENE = 3
    CURRENT_TASK = 1
    CURRENT_DEMO = 3

    # Define los objetos de la escena que vas a usar
    mis_objetos_manuales = {
        'bottle03': 'grasp', 
        'bottle04': 'grasp', 
        'wooden_crate': 'drop'
    }
    
    """ mis_objetos_manuales = {
        'banana': 'grasp', 
        'bucket_cap': 'grasp', 
        'bucket': 'drop'
    } """
    
    chain_gp_pathwise_segments(
        scene=CURRENT_SCENE, 
        task=CURRENT_TASK, 
        demo=CURRENT_DEMO, 
        manual_objects=mis_objetos_manuales,
        robot_id=franka_robot,
        ee_idx=11 
    )
    
    p.disconnect()