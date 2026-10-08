import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import time
import math

import pybullet as p
import pybullet_data

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates

# Librería local GMP
from ProGP import ProGpMp

# Entorno Franka Kitchen
from frankaKitchenPybullet.env import Environment
from frankaKitchenPybullet.robot import Robot
import frankaKitchenPybullet.config

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'cyan']



def draw_debug_target(pos, label, is_grasp=True):
    color = [0, 1, 0] if is_grasp else [1, 0, 0] 
    length = 0.04
    p.addUserDebugLine([pos[0]-length, pos[1], pos[2]], [pos[0]+length, pos[1], pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1]-length, pos[2]], [pos[0], pos[1]+length, pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1], pos[2]-length], [pos[0], pos[1], pos[2]+length], color, 3)
    p.addUserDebugText(label, [pos[0], pos[1], pos[2]+0.05], textColorRGB=color, textSize=1.2)

def unwrap_quaternions(quats):
    unwrapped = np.empty_like(quats)
    unwrapped[0] = quats[0]
    for i in range(1, len(quats)):
        if np.dot(unwrapped[i-1], quats[i]) < 0:
            unwrapped[i] = -quats[i]
        else:
            unwrapped[i] = quats[i]
    return unwrapped

def load_segment_data(base_dir, task_name, demo_idx, segment_idx, gap=5):
    folder_path = os.path.join(base_dir, task_name)
    file_name = f"demo_{demo_idx}_segmento_{segment_idx}.txt" 
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path): raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    data_full = np.loadtxt(file_path)
    data = data_full[::gap, :]
    dt = 0.01 
    num_points = data.shape[0]
    time_arr = (np.arange(num_points) * dt).reshape(-1, 1)
    Y_pos = data[:, 0:3]
    Y_quat = data[:, 3:7]
    Y_quat = unwrap_quaternions(Y_quat)
    return time_arr, Y_pos, Y_quat, dt

# ============================================================================
# FUNCIÓN DE CÁLCULO DE POSICIÓN
# ============================================================================
def get_object_grasp_position(obj_key, obj_data):
    if 'pos' in obj_data:
        return np.array(obj_data['pos'], dtype=float)
        
    b_id = obj_data['id']
    l_idx = obj_data['link']
    
    if l_idx == -1:
        pos, ori = p.getBasePositionAndOrientation(b_id)
    else:
        state = p.getLinkState(b_id, l_idx)
        pos, ori = state[4], state[5]
        
    t_pos = np.array(pos)
    
    if obj_key == 'kettle':
        t_pos[2] += 0.12
    elif obj_key == 'hinge':
        t_pos[0] -= 0.13
        t_pos[1] -= 0.35
    elif obj_key == 'slide':
        t_pos[0] -= 0.10  
    elif 'bottomknob' in obj_key:
        t_pos[0] -= 0.15
    elif 'topknob' in obj_key:
        t_pos[0] -= 0.15
        #t_pos[1] -= 0.15
    elif 'microwave' in obj_key:
        t_pos[1] -= 0.35
        
    action_type = obj_data.get('action')
    if action_type == 'drop':
        t_pos[2] += 0.1
    elif action_type == 'grasp' and obj_key != 'kettle': 
        t_pos[2] -= 0.02
        
    return t_pos

# ============================================================================
# CLASE CONTROLADOR: VISUALIZADOR 3D INTERACTIVO Y PLANIFICADOR DE AGARRE
# ============================================================================
class KitchenGPController:
    def __init__(self, segments_info, all_prior, env, robot, ee_idx, scene_objects, initial_qpos, movable_joints):
        self.segments_info = segments_info
        self.all_prior = all_prior
        self.env = env
        self.robot = robot
        self.ee_idx = ee_idx
        self.scene_objects = scene_objects 
        self.initial_qpos = initial_qpos
        self.movable_joints = movable_joints
        
        self.current_step = 0
        self.full_trajectory_pos = None
        self.full_trajectory_quat = None

        self.gripper_width = 0.04           
        self.grasp_state = "IDLE"           
        self.pause_counter = 0
        self.grasped_objects = set()        
        self.dropped_objects = set() 
        self.current_target_obj = None
        self.grasp_constraint = None 
        
        self.auto_release_counter = 0

        self.original_knob_link = 15  
        for k, data in self.scene_objects.items():
            if 'knob' in k:
                self.original_knob_link = data['link']
                break

        self.points = []
        self.point_mapping = [] 
        for seg_idx, info in enumerate(self.segments_info):
            for v_idx, pt in enumerate(info['vias_cond_pos']):
                self.points.append(pt)
                self.point_mapping.append((seg_idx, v_idx))
                
        self.points = np.array(self.points, float)
        
        self.shared_groups = []
        visited = set()
        for i in range(len(self.points)):
            if i in visited: continue
            shared_idxs = [j for j in range(len(self.points)) if np.linalg.norm(self.points[j] - self.points[i]) < 1e-3]
            visited.update(shared_idxs)
            self.shared_groups.append({
                'idxs': shared_idxs,
                'orig_pos': self.points[i].copy()
            })

        self.lines = []
        
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)
        self.ax.set_title("Pathwise Conditioning Dinámico (Pulsa 'R' en PyBullet)", fontsize=14)

        for i in range(len(segments_info)):
            self.ax.plot(all_prior[i][:, 0], all_prior[i][:, 1], all_prior[i][:, 2], '--', linewidth=2, c='grey', alpha=0.4)
            line, = self.ax.plot(segments_info[i]['mean_corrected_pos'][:, 0], segments_info[i]['mean_corrected_pos'][:, 1], segments_info[i]['mean_corrected_pos'][:, 2], '-', linewidth=3, c=colors[i % len(colors)])
            self.lines.append(line)

        scatter_pts = []
        for info in self.segments_info:
            for v_idx, pt in enumerate(info['vias_cond_pos']):
                if v_idx in info['obj_names'] and info['obj_names'][v_idx] not in ["Start", "End"]:
                    scatter_pts.append(pt)
        scatter_pts = np.array(scatter_pts, float)
        
        self.scatter = None
        if len(scatter_pts) > 0:
            self.scatter = self.ax.scatter(scatter_pts[:,0], scatter_pts[:,1], scatter_pts[:,2], s=150, c='orange', marker='D')

        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='orange', marker='D', linestyle='None', markersize=10, label='Via-points'))
        self.ax.legend(handles=handles, fontsize=10, loc='best')

        self._build_full_trajectory()
        
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self.step_simulation)
        self.timer.start()

    def _build_full_trajectory(self):
        full_pos = []
        full_quat = []
        self.action_schedule = {}  
        current_offset = 0

        for info in self.segments_info:
            full_pos.append(info['mean_corrected_pos'])
            full_quat.append(info['mean_corrected_quat']) 
            
            test_x = info['test_x']
            via_t_cond = info['via_t_cond']
            obj_names = info['obj_names']
            
            for v_idx, t in enumerate(via_t_cond):
                if v_idx in obj_names:
                    label = obj_names[v_idx]
                    local_idx = int(np.argmin(np.abs(test_x.flatten() - t)))
                    global_idx = current_offset + local_idx
                    
                    if 'drop_point_' in label:
                        target_name = label.replace('drop_point_', '')
                        self.action_schedule[global_idx] = {"action": "DROP", "target": target_name}
                    else:
                        self.action_schedule[global_idx] = {"action": "GRASP", "target": label}
                        
            current_offset += len(info['mean_corrected_pos'])
            
        self.full_trajectory_pos = np.vstack(full_pos)
        self.full_trajectory_quat = np.vstack(full_quat)

    def recalculate_trajectory(self):
        print("\n=======================================================")
        print("[!] TECLA 'R' PULSADA: RECALCULANDO ENTORNO DINÁMICO")
        print("=======================================================")
        
        p.removeAllUserDebugItems()

        burner_links = [18, 20, 22, 24]
        knob_links = [6, 9, 12, 15]
        
        for knob in knob_links:
            p.resetJointState(self.env.kitchen_id, knob, 0.0)
            p.setJointMotorControl2(self.env.kitchen_id, knob, p.POSITION_CONTROL, targetPosition=0.0, force=50)

        idx = np.random.randint(0, 4)
        chosen_burner = burner_links[idx]
        chosen_knob = knob_links[idx]
        
        burner_pos = p.getLinkState(self.env.kitchen_id, chosen_burner)[0]

        if 'manual_drop' in self.scene_objects:
            self.scene_objects['manual_drop']['pos'] = [burner_pos[0], burner_pos[1], burner_pos[2] + 0.20]

        
        pos_bases_relativas = {
            6:  np.array([0.5459, 0.1248, 1.6698]), 
            9:  np.array([0.5459, 0.1048, 1.6698]), 
            12: np.array([0.5459, 0.1248, 1.7553]), 
            15: np.array([0.5459, 0.1048, 1.7553])  
        }
        
        offset_knob = pos_bases_relativas[chosen_knob] - pos_bases_relativas[self.original_knob_link]
        
        if chosen_knob in [12, 15]:
            new_knob_key = 'topknob'
        elif chosen_knob in [6, 9]:
            new_knob_key = 'bottomknob'
        else:
            new_knob_key = 'bottomknob'

        old_keys = [k for k in self.scene_objects.keys() if 'knob' in k]
        for k in old_keys:
            data = self.scene_objects.pop(k)
            data['link'] = chosen_knob
            self.scene_objects[new_knob_key] = data

        available_burners = [b for b in burner_links if b != chosen_burner]
        kettle_burner_link = np.random.choice(available_burners)
        kettle_burner_pos = p.getLinkState(self.env.kitchen_id, int(kettle_burner_link))[0]
        
        dx = np.random.uniform(-0.02, 0.02)
        dy = np.random.uniform(-0.02, 0.02)
        new_kettle_pos = [kettle_burner_pos[0] + dx, kettle_burner_pos[1] + dy, kettle_burner_pos[2] + 0.02] 
        
        p.resetBasePositionAndOrientation(self.env.kettle_id, new_kettle_pos, p.getQuaternionFromEuler([0,0,0]))
        p.resetBaseVelocity(self.env.kettle_id, [0,0,0], [0,0,0])

        hinge_pos = np.random.uniform(0.0, 0.6)  
        slide_pos = np.random.uniform(0.0, 0.25) 
        p.resetJointState(self.env.kitchen_id, 46, hinge_pos)
        p.setJointMotorControl2(self.env.kitchen_id, 46, p.POSITION_CONTROL, targetPosition=hinge_pos, force=500)
        p.resetJointState(self.env.kitchen_id, 41, slide_pos)
        p.setJointMotorControl2(self.env.kitchen_id, 41, p.POSITION_CONTROL, targetPosition=slide_pos, force=500)

        # 1. ACTUALIZAR LAS POSICIONES CONDICIONALES EXACTAS
        for group in self.shared_groups:
            orig_pos = group['orig_pos']
            
            group_labels = []
            for idx in group['idxs']:
                seg_idx, v_idx = self.point_mapping[idx]
                l = self.segments_info[seg_idx]['obj_names'].get(v_idx, "")
                if l: group_labels.append(l)
                
            is_knob = any('knob' in l for l in group_labels)
            is_drop = any('drop' in l for l in group_labels)
            
            if is_knob:
                new_pos = orig_pos + offset_knob
                for idx in group['idxs']:
                    seg_idx, v_idx = self.point_mapping[idx]
                    if self.segments_info[seg_idx]['obj_names'].get(v_idx, "") != "":
                        self.segments_info[seg_idx]['obj_names'][v_idx] = new_knob_key
                #draw_debug_target(new_pos, f"AGARRE_{new_knob_key}", is_grasp=True)
                
            elif is_drop:
                new_pos = np.array([burner_pos[0], burner_pos[1], burner_pos[2] + 0.20])
                #draw_debug_target(new_pos, "DROP_ZONA", is_grasp=False)
                
            else:
                dyn_label = None
                for l in group_labels:
                    if l in self.scene_objects:
                        dyn_label = l
                        break
                if dyn_label:
                    obj_data = self.scene_objects[dyn_label]
                    new_pos = get_object_grasp_position(dyn_label, obj_data)
                    #draw_debug_target(new_pos, f"REACH {dyn_label}", is_grasp=True)
                else:
                    new_pos = orig_pos.copy() 
            
            for idx in group['idxs']:
                seg_idx, v_idx = self.point_mapping[idx]
                self.segments_info[seg_idx]['vias_cond_pos'][v_idx] = new_pos

        # 2. RECOMPUTAR Y [APLICAR EL HACK GUARRO] AL PRIOR
        for seg_idx, info in enumerate(self.segments_info):
            # Restauramos el Prior Virginal
            mean_gmp_arr = info['mean_gmp_arr_orig'].copy()
            
            """ # --- MAGIA GUARRA DE LA RAMPA ---
            ramp = np.zeros((len(info['test_x']), 1))
            applied_ramp = False
            
            for v_idx, label in info['obj_names'].items():
                if 'knob' in label:
                    target_t = info['via_t_cond'][v_idx]
                    local_idx = int(np.argmin(np.abs(info['test_x'].flatten() - target_t)))
                    
                    # Generamos un triángulo/rampa adaptativo: 0% al inicio, 100% en el knob, 0% al final
                    # (Esto evita teletransportes bruscos si el knob está al principio o en medio)
                    if local_idx > 0:
                        ramp[:local_idx+1, 0] = np.linspace(0.0, 1.0, local_idx + 1)
                    if local_idx < len(info['test_x']) - 1:
                        ramp[local_idx:, 0] = np.linspace(1.0, 0.0, len(info['test_x']) - local_idx)
                        
                    applied_ramp = True
                    break 
            
            if applied_ramp:
                print(f"   [!] Inyectando offset guarro (Rampa adaptativa) al segmento {seg_idx+1}")
                # Modificamos toda la matriz del movimiento base hacia la zona nueva
                mean_gmp_arr = mean_gmp_arr + (ramp * offset_knob).T
            # -------------------------------- """
            
            """ # --- MAGIA GUARRA: OFFSET DEL TIRÓN ---
            apply_offset = False
            
            for v_idx, label in info['obj_names'].items():
                if 'knob' in label:
                    apply_offset = True
                    break 
            
            if apply_offset:
                print(f"   [!] Inyectando offset DEL TIRÓN al segmento {seg_idx+1}")
                
                # offset_knob tiene forma (3,), lo convertimos a (3, 1) con reshape 
                # para que NumPy lo sume correctamente a mean_gmp_arr que es (3, N)
                mean_gmp_arr = mean_gmp_arr + offset_knob.reshape(3, 1) """
            # --------------------------------
            
            # Guardamos el Prior Modificado para que GPFlow haga el Exact Inference sobre él
            info['mean_gmp_arr'] = mean_gmp_arr
            info['F_prior_full'] = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

            kernel = info['kernel']
            F_prior_full = info['F_prior_full']
            test_x = info['test_x']

            X_obs = tf.convert_to_tensor(info['via_t_cond'].reshape(-1, 1), dtype=default_float())
            Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(info['vias_cond_pos'], dtype=default_float()), axis=0)
            
            obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in info['via_t_cond']]
            F_obs_pos = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 
            
            # Pathwise Conditioning
            dF_fn_pos = updates.exact(kernel, X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
            X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 
            mean_corrected_pos = np.squeeze((F_prior_full + dF_fn_pos(X_test)).numpy())

            info['mean_corrected_pos'] = mean_corrected_pos
            self.lines[seg_idx].set_data_3d(mean_corrected_pos[:,0], mean_corrected_pos[:,1], mean_corrected_pos[:,2])

        self._build_full_trajectory()

        new_scatter = []
        for info in self.segments_info:
            for v_idx, pt in enumerate(info['vias_cond_pos']):
                if v_idx in info['obj_names'] and info['obj_names'][v_idx] not in ["Start", "End"]:
                    new_scatter.append(pt)
        if len(new_scatter) > 0 and self.scatter is not None:
            new_scatter = np.array(new_scatter)
            self.scatter._offsets3d = (new_scatter[:,0], new_scatter[:,1], new_scatter[:,2])
        
        self.fig.canvas.draw_idle()

        self.current_step = 0
        self.grasp_state = "IDLE"
        self.auto_release_counter = 0
        self.grasped_objects.clear()
        self.dropped_objects.clear()
        self.current_target_obj = None
        self.gripper_width = 0.04
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

        print(" -> Eliminando robot anterior y cargando uno nuevo desde cero...")
        p.removeBody(self.robot.id)
        
        self.robot = Robot()
        self.movable_joints = [j for j in range(p.getNumJoints(self.robot.id)) if p.getJointInfo(self.robot.id, j)[2] in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]]

        print("[Recálculo] Posicionando al robot en su estado de descanso normal...")
        rest_qpos = [0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.04, 0.04]
        for i, joint_idx in enumerate(self.movable_joints):
            if i < len(rest_qpos): 
                p.resetJointState(self.robot.id, joint_idx, rest_qpos[i])
                
        for _ in range(50): 
            self.env.update()

        time.sleep(2.0)

        print("[Recálculo] Viajando a la postura inicial de la tarea de forma suave...")
        for i, joint_idx in enumerate(self.movable_joints):
            if i < len(self.initial_qpos):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot.id, jointIndex=joint_idx, controlMode=p.POSITION_CONTROL,
                    targetPosition=self.initial_qpos[i], force=300, maxVelocity=1.5
                )
                
        for _ in range(150): 
            self.env.update()

    def step_simulation(self):
        keys = p.getKeyboardEvents()
        if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
            self.recalculate_trajectory()
            return
        
        if self.full_trajectory_pos is not None and len(self.full_trajectory_pos) > 0:
            target_pos = self.full_trajectory_pos[self.current_step]
            target_quat = self.full_trajectory_quat[self.current_step]
            
            euler = p.getEulerFromQuaternion(target_quat)
            target_quat_adjusted = p.getQuaternionFromEuler([euler[0], euler[1], euler[2]])
            
            if self.grasp_state == "IDLE":
                if hasattr(self, 'auto_release_counter') and self.auto_release_counter > 0:
                    self.auto_release_counter -= 1
                    if self.auto_release_counter <= 0:
                        self.grasp_state = "RELEASING"
                        self.pause_counter = 15
                        self.gripper_width = 0.04 
                
                if self.grasp_state == "IDLE" and self.current_step in self.action_schedule:
                    action_info = self.action_schedule[self.current_step]
                    
                    if action_info["action"] == "GRASP" and len(self.grasped_objects) == 0:
                        obj_name = action_info["target"]
                        if obj_name in ['kettle', 'hinge', 'slide'] and obj_name in self.scene_objects and 'id' in self.scene_objects[obj_name]:
                            self.grasp_state = "WAITING"
                            self.pause_counter = 20
                            self.current_target_obj = obj_name
                            
                    elif action_info["action"] == "DROP" and len(self.grasped_objects) > 0:
                        self.grasp_state = "RELEASING"
                        self.pause_counter = 15
                        self.gripper_width = 0.04

            elif self.grasp_state == "WAITING":
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    self.gripper_width = 0.0 
                    self.grasp_state = "CLOSING"
                    self.pause_counter = 15 
                    
            elif self.grasp_state == "CLOSING":
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    
                    if self.current_target_obj == 'kettle':
                        obj_info = self.scene_objects[self.current_target_obj]
                        obj_id = obj_info['id']
                        obj_link = obj_info['link']
                        
                        if obj_link == -1:
                            obj_pos, obj_ori = p.getBasePositionAndOrientation(obj_id)
                        else:
                            state = p.getLinkState(obj_id, obj_link)
                            obj_pos, obj_ori = state[4], state[5]

                        ee_state = p.getLinkState(self.robot.id, self.ee_idx)
                        ee_pos, ee_ori = ee_state[4], ee_state[5] 
                        
                        inv_ee_pos, inv_ee_ori = p.invertTransform(ee_pos, ee_ori)
                        snap_rel_pos, rel_ori = p.multiplyTransforms(inv_ee_pos, inv_ee_ori, obj_pos, obj_ori)
                        
                        self.grasp_constraint = p.createConstraint(
                            parentBodyUniqueId=self.robot.id, parentLinkIndex=self.ee_idx, childBodyUniqueId=obj_id,
                            childLinkIndex=obj_link, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                            parentFramePosition=snap_rel_pos, childFramePosition=[0, 0, 0], parentFrameOrientation=rel_ori
                        )
                        p.changeConstraint(self.grasp_constraint, maxForce=5000)
                    
                    self.grasped_objects.add(self.current_target_obj)
                    
                    if self.current_target_obj in ['hinge', 'slide']:
                        self.auto_release_counter = 90  
                        
                    self.grasp_state = "IDLE"

            elif self.grasp_state == "RELEASING":
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    self.gripper_width = 0.04 
                    if self.grasp_constraint is not None:
                        p.removeConstraint(self.grasp_constraint)
                        self.grasp_constraint = None
                        
                    if self.current_target_obj:
                        self.dropped_objects.add(self.current_target_obj)
                        
                    self.grasped_objects.clear()
                    self.current_target_obj = None
                    self.grasp_state = "IDLE"

            joint_poses = list(p.calculateInverseKinematics(
                self.robot.id, self.ee_idx, targetPosition=target_pos, targetOrientation=target_quat_adjusted, maxNumIterations=100
            ))
            
            for _ in range(15):
                for i, joint_idx in enumerate(self.movable_joints):
                    if i < 7: target_j = joint_poses[i]
                    else: target_j = self.gripper_width 
                        
                    p.setJointMotorControl2(
                        bodyUniqueId=self.robot.id, jointIndex=joint_idx, controlMode=p.POSITION_CONTROL,
                        targetPosition=target_j, force=300, maxVelocity=5.0
                    )
                p.stepSimulation()
            
            self.env.update()
            
            if self.grasp_state == "IDLE":
                self.current_step += 1
                if self.current_step >= len(self.full_trajectory_pos):
                    self.current_step = 0
                    self.auto_release_counter = 0
                    self.gripper_width = 0.04
                    if self.grasp_constraint is not None:
                        p.removeConstraint(self.grasp_constraint)
                        self.grasp_constraint = None
                    self.grasped_objects.clear()
                    self.dropped_objects.clear()
                    self.current_target_obj = None

    def show(self):
        plt.show()

# ============================================================================
# LÓGICA PRINCIPAL: GMP + PATHWISE CONDITIONING
# ============================================================================
def chain_gp_pathwise_kitchen(env, robot, manual_objects, base_dir, task_name, demo_idx, ee_idx, initial_qpos, movable_joints):
    
    print(f"--- Procesando Pathwise Conditioning para {task_name} | Demo {demo_idx} ---")

    all_prior, segments_info = [], []
    previous_end_pos = None

    target_positions = {}
    global_min_dist = {}
    
    if manual_objects:
        for obj_name, obj_data in manual_objects.items():
            t_pos = get_object_grasp_position(obj_name, obj_data)
            target_positions[obj_name] = [t_pos]
            global_min_dist[obj_name] = float('inf')

        seg_check = 1
        while True:
            try:
                X_c, Y_c, _, _ = load_segment_data(base_dir, task_name, demo_idx, seg_check)
                for obj_name, t_pos_list in target_positions.items():
                    for t_pos in t_pos_list:
                        min_d = np.min(np.linalg.norm(Y_c - t_pos, axis=1))
                        if min_d < global_min_dist[obj_name]:
                            global_min_dist[obj_name] = min_d
            except FileNotFoundError: break
            seg_check += 1

    segment_matches = {}
    if manual_objects:
        seg_check = 1
        while True:
            try:
                X_c, Y_c, Y_q_c, _ = load_segment_data(base_dir, task_name, demo_idx, seg_check)
            except FileNotFoundError: break 
            
            segment_matches[seg_check] = []
            for obj_name, t_pos_list in target_positions.items():
                best_dist, best_idx, best_t_pos = float('inf'), -1, None
                for t_pos in t_pos_list:
                    distances = np.linalg.norm(Y_c - t_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < best_dist:
                        best_dist, best_idx, best_t_pos = distances[closest_idx], closest_idx, t_pos
                
                is_manual_drop = ('pos' in manual_objects[obj_name])
                threshold = 0.85 if is_manual_drop else 0.45
                
                if best_dist <= global_min_dist[obj_name] + 0.08 and best_dist < threshold:
                    segment_matches[seg_check].append({
                        'obj_name': obj_name, 'time': X_c[best_idx, 0], 'target_pos': best_t_pos,
                        'target_quat': Y_q_c[best_idx, :].copy(), 'action_type': manual_objects[obj_name]['action']
                    })
            seg_check += 1

    segment_idx = 1
    while True:
        try:
            X, Y_pos, Y_quat, dt = load_segment_data(base_dir, task_name, demo_idx, segment_idx)
            size = X.shape[0]
        except FileNotFoundError: break
            
        print(f">> Aplicando GMP y Pathwise al Segmento {segment_idx}...")
        
        target_t = X[-1, 0]
        orig_start_pos, orig_end_pos = Y_pos[0, :], Y_pos[-1, :]
        via_t = np.array([X[0, 0], target_t]).reshape(-1, 1)
        
        curr_start_pos = orig_start_pos if previous_end_pos is None else previous_end_pos
        curr_end_pos = orig_end_pos
        vias_orig_pos = np.array([orig_start_pos, orig_end_pos])
        
        gp_pos = ProGpMp(X, Y_pos, via_t, vias_orig_pos, dim=3, demos=1, size=size, observation_noise=0.005)
        gp_pos.BlendedGpMp(gp_pos.ProGP)
        
        test_x = np.linspace(X[0, 0], X[-1, 0], size * 4).reshape(-1, 1)
        mean_gmp_pos, _ = gp_pos.predict_BlendedPos(test_x)

        orig_time = X.flatten()
        new_time = test_x.flatten()
        Y_quat_interp = np.zeros((len(new_time), 4))
        for i in range(4): Y_quat_interp[:, i] = np.interp(new_time, orig_time, Y_quat[:, i])
        Y_quat_interp /= np.linalg.norm(Y_quat_interp, axis=1, keepdims=True)

        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        kernel = kernels.SeparateIndependent([cls(lengthscales=0.05) for cls in BaseClasses])
        mean_gmp_arr = np.vstack(mean_gmp_pos) 
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

        points_dict = { via_t[0, 0]: (curr_start_pos, "Start"), target_t: (curr_end_pos, "End") }

        if manual_objects and segment_idx in segment_matches:
            for match in segment_matches[segment_idx]:
                closest_time = match['time']
                target_pos = match['target_pos']
                
                if match['action_type'] == 'drop': 
                    virtual_name = f"drop_point_{match['obj_name']}"
                    #draw_debug_target(target_pos, f"DROP {match['obj_name']}", is_grasp=False)
                else: 
                    virtual_name = match['obj_name']
                    texto = f"GRASP {virtual_name}" if virtual_name in ['kettle', 'hinge', 'slide'] else f"REACH {virtual_name}"
                    #draw_debug_target(target_pos, texto, is_grasp=True)
                
                if abs(closest_time - via_t[0, 0]) <= 0.05: points_dict[via_t[0, 0]] = (target_pos, virtual_name)
                elif abs(closest_time - target_t) <= 0.05: points_dict[target_t] = (target_pos, virtual_name)
                else:
                    while closest_time in points_dict: closest_time += 1e-4  
                    points_dict[closest_time] = (target_pos, virtual_name)

        previous_end_pos = points_dict[target_t][0]
        points_info_list = sorted([(t, val[0], val[1]) for t, val in points_dict.items()], key=lambda x: x[0])

        via_t_cond = np.array([p[0] for p in points_info_list])
        vias_cond_pos = np.array([p[1] for p in points_info_list])
        obj_names_map = {idx: p[2] for idx, p in enumerate(points_info_list) if p[2] not in ["Start", "End"]}

        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(vias_cond_pos, dtype=default_float()), axis=0)
        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        F_obs_pos = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 
        
        dF_fn_pos = updates.exact(kernel, X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
        X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 
        mean_corrected_pos = np.squeeze((F_prior_full + dF_fn_pos(X_test)).numpy())

        all_prior.append(np.squeeze(mean_gmp_arr.T))

        segments_info.append({
            'kernel': kernel, 'F_prior_full': F_prior_full, 'via_t_cond': via_t_cond, 
            'vias_cond_pos': vias_cond_pos.copy(), 
            'vias_cond_pos_orig': vias_cond_pos.copy(), 
            'mean_gmp_arr': mean_gmp_arr, 
            'mean_gmp_arr_orig': mean_gmp_arr.copy(), # <--- MAGIA: Guardamos el original intacto 
            'test_x': test_x, 
            'obj_names': obj_names_map, 'mean_corrected_pos': mean_corrected_pos, 'mean_corrected_quat': Y_quat_interp 
        })
        segment_idx += 1

    if len(segments_info) > 0:
        print("\n>> Abriendo Visor Interactivo y reproduciendo en PyBullet...")
        plotter = KitchenGPController(segments_info, all_prior, env, robot, ee_idx, manual_objects, initial_qpos, movable_joints)
        plotter.show()
    else: print("[!] No se encontraron segmentos para procesar.")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")
    
    print("Cargando el entorno de Franka Kitchen...")
    env = Environment()
    env.load()
    
    burner_pos = p.getLinkState(env.kitchen_id, 20)[0]
    
    kettle_urdf_path = "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/LongTaskRepo/frankaKitchenPybullet/kitchen_assets/item_assets/kettle.urdf"
    kettle_pos = [burner_pos[0]-0.15, burner_pos[1], burner_pos[2] + 0.02] 
    kettle_ori = p.getQuaternionFromEuler([0, 0, 0])
    
    if not hasattr(env, 'kettle_id') or env.kettle_id is None:
        env.kettle_id = p.loadURDF(kettle_urdf_path, kettle_pos, kettle_ori, useFixedBase=False, globalScaling=0.7)
    
    robot = Robot()
    movable_joints_main = [j for j in range(p.getNumJoints(robot.id)) if p.getJointInfo(robot.id, j)[2] in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]]

    print("\n[Inicialización] Posicionando al robot en su estado de descanso normal...")
    rest_qpos = [0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.04, 0.04]
    for i, joint_idx in enumerate(movable_joints_main):
        if i < len(rest_qpos): p.resetJointState(robot.id, joint_idx, rest_qpos[i])
            
    for _ in range(50): env.update()

    print("[Inicialización] Viajando a la postura inicial de la tarea de forma suave...")
    initial_qpos = [
        1.5161467790603638,
        0.03982258960604668,
        0.03999759256839752,
        0.12671640515327454,
        -1.7634446620941162,
        1.8679695129394531,
        -2.497931718826294,
        0.3437056839466095,
        2.3736004988339285
    ]
    
    for i, joint_idx in enumerate(movable_joints_main):
        if i < len(initial_qpos):
            p.setJointMotorControl2(
                bodyUniqueId=robot.id, jointIndex=joint_idx, controlMode=p.POSITION_CONTROL,
                targetPosition=initial_qpos[i], force=300, maxVelocity=1.5
            )
            
    for _ in range(150): env.update()
    print("[Inicialización] Postura inicial de la tarea alcanzada.\n")

    ee_link_index = -1
    for j in range(p.getNumJoints(robot.id)):
        if p.getJointInfo(robot.id, j)[12].decode('utf-8') == 'panda_link8':
            ee_link_index = j
            break
    if ee_link_index == -1: ee_link_index = 8 

    CARPETA_SEGMENTOS = "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/LongTaskRepo/Segmentation-and-Grouping-Model/SegmentationProb/scripts/KitchenFranka_SegmentsFolder/" 
    TAREA_A_PROCESAR = "postcorl_microwave_kettle_switch_hinge" 
    DEMO_SELECCIONADA = 1

    KITCHEN_OBJECTS_MAP = {
        'microwave': {'action': 'grasp', 'id': env.kitchen_id, 'link': 54}, 
        'kettle': {'action': 'grasp', 'id': env.kettle_id, 'link': -1},     
        'light': {'action': 'grasp', 'id': env.kitchen_id, 'link': 29},     
        'bottomknob': {'action': 'grasp', 'id': env.kitchen_id, 'link': 12}, 
        'topknob': {'action': 'grasp', 'id': env.kitchen_id, 'link': 6},    
        'hinge': {'action': 'grasp', 'id': env.kitchen_id, 'link': 46},      
        'slide': {'action': 'grasp', 'id': env.kitchen_id, 'link': 41},      
        'burner': {'action': 'grasp', 'id': env.kitchen_id, 'link': 18}      
    }

    OBJETOS_A_CONDICIONAR = {}
    for word in TAREA_A_PROCESAR.split('_'):
        if word in KITCHEN_OBJECTS_MAP:
            OBJETOS_A_CONDICIONAR[word] = KITCHEN_OBJECTS_MAP[word]
            
    OBJETOS_A_CONDICIONAR['manual_drop'] = {
        'action': 'drop', 
        'pos': [0.27, 0.23, 1.40] 
    }
            
    print(f"Objetos detectados para la tarea: {list(OBJETOS_A_CONDICIONAR.keys())}")

    chain_gp_pathwise_kitchen(
        env=env,
        robot=robot,
        manual_objects=OBJETOS_A_CONDICIONAR,
        base_dir=CARPETA_SEGMENTOS,
        task_name=TAREA_A_PROCESAR,
        demo_idx=DEMO_SELECCIONADA,
        ee_idx=ee_link_index,
        initial_qpos=initial_qpos,
        movable_joints=movable_joints_main
    )
    
    p.disconnect()