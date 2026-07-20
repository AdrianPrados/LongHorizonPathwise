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
# HELPER: CUATERNIONES A MATRIZ DE ROTACIÓN (PARA DIBUJAR DIRECCIÓN DE AGARRE)
# ============================================================================
def quat_to_rotmat(q):
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-8:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

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
        self.quiver_artists = []
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
            for v_idx, pt in enumerate(info['vias_cond_pos']):
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
        self.ax.set_title("Visualizador Pasivo de Trayectoria 6D\n(Presiona la tecla 'P' en PyBullet para generar gráficas)", fontsize=14)

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

        # Dibuja las direcciones de agarre (orientación del end-effector)
        self.update_quivers()

        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='orange', marker='D', linestyle='None', markersize=10, label='Via-point (Objeto)'))
        handles.append(mlines.Line2D([], [], color='red', marker='D', linestyle='None', markersize=10, label='Virtual Drop Point'))
        handles.append(mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Nodo Base'))
        handles.append(mlines.Line2D([], [], color='darkred', marker='>', linestyle='None', markersize=10, label='Dirección Agarre (Eje Z)'))
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
            init_quat = self.segments_info[seg_idx]['vias_cond_quat'][v_idx]
            
            # Normalizamos antes de extraer Euler para evitar errores
            init_quat = init_quat / np.linalg.norm(init_quat)
            init_rpy = p.getEulerFromQuaternion(init_quat)
            
            sx = p.addUserDebugParameter(f"{label} [X]", init_pos[0]-0.6, init_pos[0]+0.6, init_pos[0])
            sy = p.addUserDebugParameter(f"{label} [Y]", init_pos[1]-0.6, init_pos[1]+0.6, init_pos[1])
            sz = p.addUserDebugParameter(f"{label} [Z]", 0.1, 1.5, init_pos[2])
            
            sr = p.addUserDebugParameter(f"{label} [R]", -math.pi, math.pi, init_rpy[0])
            sp = p.addUserDebugParameter(f"{label} [P]", -math.pi, math.pi, init_rpy[1])
            syaw = p.addUserDebugParameter(f"{label} [Yaw]", -math.pi, math.pi, init_rpy[2])
            
            self.slider_groups.append({'idxs': shared_idxs, 'sliders': (sx, sy, sz, sr, sp, syaw), 'label': label})
            self.last_slider_reads.append(np.concatenate([init_pos, init_rpy]))

        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self.step_simulation)
        self.timer.start()

    def update_quivers(self):
        # Limpia las flechas anteriores
        for q in self.quiver_artists:
            try:
                q.remove()
            except:
                pass
        self.quiver_artists = []
        
        # Genera las nuevas flechas de orientación
        for seg_idx, info in enumerate(self.segments_info):
            mean_corr = info.get('mean_corrected', self.all_corrected[seg_idx])
            
            # Puntos obligatorios: Inicio (Intercambio), Medio (Intermedio), Final (Objeto/Drop)
            indices = [0, len(mean_corr)//2, len(mean_corr)-1]
            
            for idx in indices:
                pt = mean_corr[idx]
                pos = pt[:3]
                quat = pt[3:7]
                
                # Extraemos la matriz de rotación del cuaternión
                rot = quat_to_rotmat(quat)
                # El eje Z local del efector final suele ser la dirección de aproximación/agarre
                z_axis = rot[:, 2] 
                
                # Pintamos el vector de agarre
                q_artist = self.ax.quiver(
                    pos[0], pos[1], pos[2], 
                    z_axis[0], z_axis[1], z_axis[2], 
                    length=0.12, color='darkred', linewidth=2.5, arrow_length_ratio=0.2
                )
                self.quiver_artists.append(q_artist)

    def plot_full_trajectory(self):
        t_full = []
        prior_pos_full = []
        corr_pos_full = []
        prior_quat_full = []
        corr_quat_full = []

        via_t_full = []
        via_pos_full = []
        via_quat_full = []

        current_t_offset = 0.0

        for i, info in enumerate(self.segments_info):
            test_x = info['test_x'].flatten()
            dt_segment = test_x[-1] - test_x[0]

            shifted_t = test_x - test_x[0] + current_t_offset
            t_full.append(shifted_t)

            prior = self.all_prior[i]
            corr = info['mean_corrected']

            prior_pos_full.append(prior[:, :3])
            prior_quat_full.append(prior[:, 3:7])
            corr_pos_full.append(corr[:, :3])
            corr_quat_full.append(corr[:, 3:7])

            via_t = info['via_t_cond'].flatten()
            shifted_via_t = via_t - test_x[0] + current_t_offset
            via_t_full.append(shifted_via_t)
            via_pos_full.append(info['vias_cond_pos'])
            via_quat_full.append(info['vias_cond_quat'])

            current_t_offset += dt_segment + (test_x[1] - test_x[0])

        t_full = np.concatenate(t_full)
        prior_pos_full = np.vstack(prior_pos_full)
        corr_pos_full = np.vstack(corr_pos_full)
        prior_quat_full = np.vstack(prior_quat_full)
        corr_quat_full = np.vstack(corr_quat_full)

        via_t_full = np.concatenate(via_t_full)
        via_pos_full = np.vstack(via_pos_full)
        via_quat_full = np.vstack(via_quat_full)

        delta_pos_full = corr_pos_full - prior_pos_full
        delta_quat_full = corr_quat_full - prior_quat_full

        delta_via_pos_full = []
        delta_via_quat_full = []
        for i, t in enumerate(via_t_full):
            idx = np.argmin(np.abs(t_full - t))
            delta_via_pos_full.append(via_pos_full[i] - prior_pos_full[idx])
            delta_via_quat_full.append(via_quat_full[i] - prior_quat_full[idx])
        delta_via_pos_full = np.array(delta_via_pos_full)
        delta_via_quat_full = np.array(delta_via_quat_full)

        # GRAFICAS PARA POSICIÓN (1 Fila, 3 Columnas: X, Y, Z superpuestas)
        fig_pos, axes_pos = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        fig_pos.suptitle('Long-Horizon Task - Position Pathwise Conditioning', fontsize=18, fontweight='bold')
        cols_titles = ['Prior GMP Position', 'Pathwise Update (Delta)', 'Corrected Path Position']
        labels_pos = ['X', 'Y', 'Z']
        colors_pos = ['red', 'green', 'blue']

        for col in range(3):
            axes_pos[col].set_title(cols_titles[col], fontsize=14)
            axes_pos[col].set_xlabel('Time [s]', fontsize=14)
            axes_pos[col].grid(True, alpha=0.4)
            for t_sep in np.unique(via_t_full):
                axes_pos[col].axvline(t_sep, color='black', linestyle=':', alpha=0.3)

        axes_pos[0].set_ylabel('Position [m]', fontsize=14)
        axes_pos[1].set_ylabel('Delta [m]', fontsize=14)
        axes_pos[2].set_ylabel('Position [m]', fontsize=14)

        for dim in range(3):
            # Col 0: Prior
            axes_pos[0].plot(t_full, prior_pos_full[:, dim], color=colors_pos[dim], linewidth=3, label=f'Prior {labels_pos[dim]}')
            # Col 1: Delta
            axes_pos[1].plot(t_full, delta_pos_full[:, dim], color=colors_pos[dim], linewidth=3, label=f'$\Delta$ {labels_pos[dim]}')
            axes_pos[1].scatter(via_t_full, delta_via_pos_full[:, dim], c=colors_pos[dim], marker='x', s=100, zorder=5)
            # Col 2: Corrected
            axes_pos[2].plot(t_full, corr_pos_full[:, dim], color=colors_pos[dim], linewidth=3, label=f'Corr {labels_pos[dim]}')
            axes_pos[2].scatter(via_t_full, via_pos_full[:, dim], c=colors_pos[dim], marker='x', s=100, zorder=5)

        for col in range(3):
            axes_pos[col].legend(loc='best', frameon=False)

        plt.tight_layout()

        # GRAFICAS PARA ORIENTACIÓN (1 Fila, 3 Columnas: qx, qy, qz, qw superpuestas)
        fig_quat, axes_quat = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        fig_quat.suptitle('Long-Horizon Task - Orientation Pathwise Conditioning (Quaternions)', fontsize=18, fontweight='bold')
        cols_titles_q = ['Prior GMP Orientation', 'Pathwise Update (Delta)', 'Corrected Path Orientation']
        labels_quat = ['qx', 'qy', 'qz', 'qw']
        colors_quat = ['red', 'green', 'blue', 'orange']

        for col in range(3):
            axes_quat[col].set_title(cols_titles_q[col], fontsize=14)
            axes_quat[col].set_xlabel('Time [s]', fontsize=14)
            axes_quat[col].grid(True, alpha=0.4)
            for t_sep in np.unique(via_t_full):
                axes_quat[col].axvline(t_sep, color='black', linestyle=':', alpha=0.3)

        axes_quat[0].set_ylabel('Quaternion value', fontsize=14)
        axes_quat[1].set_ylabel('Delta', fontsize=14)
        axes_quat[2].set_ylabel('Quaternion value', fontsize=14)

        for dim in range(4):
            # Col 0: Prior
            axes_quat[0].plot(t_full, prior_quat_full[:, dim], color=colors_quat[dim], linewidth=3, label=f'Prior {labels_quat[dim]}')
            # Col 1: Delta
            axes_quat[1].plot(t_full, delta_quat_full[:, dim], color=colors_quat[dim], linewidth=3, label=f'$\Delta$ {labels_quat[dim]}')
            axes_quat[1].scatter(via_t_full, delta_via_quat_full[:, dim], c=colors_quat[dim], marker='x', s=100, zorder=5)
            # Col 2: Corrected
            axes_quat[2].plot(t_full, corr_quat_full[:, dim], color=colors_quat[dim], linewidth=3, label=f'Corr {labels_quat[dim]}')
            axes_quat[2].scatter(via_t_full, via_quat_full[:, dim], c=colors_quat[dim], marker='x', s=100, zorder=5)

        for col in range(3):
            axes_quat[col].legend(loc='best', frameon=False)

        plt.tight_layout()
        plt.show(block=True)

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

    def step_simulation(self):
        # === EVENTO DE TECLADO: PRESIONAR 'P' ===
        keys = p.getKeyboardEvents()
        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            print("\n[!] Generando gráficas de Pathwise Conditioning de la tarea completa...")
            self.plot_full_trajectory()
        # ========================================

        changed = False
        segments_to_update = set()
        
        for group_idx, group in enumerate(self.slider_groups):
            new_vals = np.array([p.readUserDebugParameter(s) for s in group['sliders']])
            new_pos = new_vals[:3]
            new_rpy = new_vals[3:]
            new_quat = p.getQuaternionFromEuler(new_rpy)
            label = group['label']
            
            if np.linalg.norm(new_vals - self.last_slider_reads[group_idx]) > 1e-3:
                changed = True
                self.last_slider_reads[group_idx] = new_vals.copy()
                
                if label in self.loaded_objects and label not in self.grasped_objects:
                    obj_id = self.loaded_objects[label]
                    offset = self.label_offsets.get(label, np.zeros(3))
                    p.resetBasePositionAndOrientation(obj_id, new_pos - offset, new_quat)
                    p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])

                for idx in group['idxs']:
                    self.points[idx] = new_pos
                    seg_idx, v_idx = self.point_mapping[idx]
                    self.segments_info[seg_idx]['vias_cond_pos'][v_idx] = new_pos
                    self.segments_info[seg_idx]['vias_cond_quat'][v_idx] = new_quat
                    segments_to_update.add(seg_idx)
        
        if changed:
            for seg_idx in segments_to_update:
                self.recompute_segment(seg_idx)
            self._build_full_trajectory()
            self.scatter._offsets3d = (self.points[:, 0], self.points[:, 1], self.points[:, 2])
            
            # Refresca las flechas de orientación dinámicamente
            self.update_quivers()
            
            self.fig.canvas.draw_idle()
            
        if self.full_trajectory is not None and len(self.full_trajectory) > 0:
            target_pos = self.full_trajectory[self.current_step][:3]
            target_ori = self.full_trajectory[self.current_step][3:7]
            
            target_ori = target_ori / np.linalg.norm(target_ori)
            
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
                            offset = self.label_offsets.get(label, np.zeros(3))
                            new_rpy = self.last_slider_reads[group_idx][3:]
                            q_ori = p.getQuaternionFromEuler(new_rpy)
                            p.resetBasePositionAndOrientation(obj_id, self.last_slider_reads[group_idx][:3] - offset, q_ori)
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
                        slider_ori = None
                        for group_idx, group in enumerate(self.slider_groups):
                            if group['label'] == obj_name:
                                slider_pos = self.last_slider_reads[group_idx][:3]
                                slider_ori = p.getQuaternionFromEuler(self.last_slider_reads[group_idx][3:])
                                break
                        
                        init_state = self.initial_object_states[obj_name]
                        if slider_pos is not None:
                            offset = self.label_offsets.get(obj_name, np.zeros(3))
                            p.resetBasePositionAndOrientation(obj_id, slider_pos - offset, slider_ori)
                        else:
                            p.resetBasePositionAndOrientation(obj_id, init_state['pos'], init_state['ori'])
                        p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])
                        
                    for j, target_val in self.robot_initial_joints.items():
                        p.resetJointState(self.robot_id, j, targetValue=target_val)

    def recompute_segment(self, seg_idx):
        info = self.segments_info[seg_idx]
        X_obs = tf.convert_to_tensor(info['via_t_cond'].reshape(-1, 1), dtype=default_float())
        obs_idx = [int(np.argmin(np.abs(info['test_x'].flatten() - t))) for t in info['via_t_cond']]
        X_test = tf.convert_to_tensor(info['test_x'], dtype=default_float())

        Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(info['vias_cond_pos'], dtype=default_float()), axis=0)
        F_obs_pos = tf.expand_dims(tf.convert_to_tensor(info['mean_gmp_arr'].T[obs_idx], dtype=default_float()), axis=0)
        dF_fn_pos = updates.exact(info['kernel'], X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
        F_cond_pos = info['F_prior_full'] + dF_fn_pos(X_test)
        mean_corrected_pos = np.squeeze(F_cond_pos.numpy())

        Y_obs_quat = tf.expand_dims(tf.convert_to_tensor(info['vias_cond_quat'], dtype=default_float()), axis=0)
        F_obs_quat = tf.expand_dims(tf.convert_to_tensor(info['mean_quat_arr'].T[obs_idx], dtype=default_float()), axis=0)
        dF_fn_quat = updates.exact(info['kernelQuat'], X_obs, Y_obs_quat, F_obs_quat, diag=0.0)
        F_cond_quat = info['F_prior_full_q'] + dF_fn_quat(X_test)
        mean_corrected_quat = np.squeeze(F_cond_quat.numpy())

        mean_corrected_full = np.hstack([mean_corrected_pos, mean_corrected_quat])
        self.segments_info[seg_idx]['mean_corrected'] = mean_corrected_full
        self.lines[seg_idx].set_data_3d(mean_corrected_pos[:,0], mean_corrected_pos[:,1], mean_corrected_pos[:,2])
        
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
    time_arr = np.arange(0, num_points * dt, dt).reshape(-1, 1)
    
    if pos_data.shape[1] >= 7:
        Y_pos = pos_data[:, :3]
        Y_quat = pos_data[:, 3:7]
    else:
        Y_pos = pos_data[:, :3]
        default_q = p.getQuaternionFromEuler([math.pi, 0, 0])
        Y_quat = np.tile(default_q, (num_points, 1))

    return time_arr, Y_pos, Y_quat, dt

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
    print(f"--- Procesando Cadena S{scene} | T{task} | Demo Base {demo} ---")

    if not os.path.exists(pkl_path): object_loc_dict = {}
    else:
        with open(pkl_path, 'rb') as f: object_loc_dict = pickle.load(f).get('object_loc_dict', {})

    previous_end_pos = None
    all_prior, all_corrected, segments_info = [], [], []

    # =========================================================================
    # FASE 0: OBTENCIÓN DINÁMICA DE COORDENADAS E INCORPORACIÓN DE ROTACIÓN
    # =========================================================================
    target_positions = {}
    global_min_dist = {}
    
    if manual_objects:
        for obj_name, action_type in manual_objects.items():
            if obj_name in loaded_objects:
                obj_id = loaded_objects[obj_name]
                aabb_min, aabb_max = p.getAABB(obj_id)
                center_x = (aabb_max[0] + aabb_min[0]) / 2.0
                center_y = (aabb_max[1] + aabb_min[1]) / 2.0
                z_top = aabb_max[2]
                _, initial_ori = p.getBasePositionAndOrientation(obj_id)
                
                target_positions[obj_name] = []
                
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
                    t_ori = np.array(initial_ori)
                    if action_type == 'drop': 
                        t_pos[2] += 0.25
                        t_ori = p.getQuaternionFromEuler([math.pi, 0, 0])
                    elif action_type == 'grasp': 
                        t_pos[2] -= 0.02
                        t_ori = p.getQuaternionFromEuler([math.pi, 0, 0])
                    
                    target_positions[obj_name].append((t_pos, t_ori))
                    
                global_min_dist[obj_name] = float('inf')
        
        seg_check = 1
        while True:
            try:
                X_c, Y_c_pos, _, _ = load_segment_data(base_dir, scene, task, demo, seg_check)
                Y_c_pos = Y_c_pos[:min(X_c.shape[0], Y_c_pos.shape[0])]
                
                for obj_name, configs in target_positions.items():
                    for t_pos, _ in configs:
                        min_d = np.min(np.linalg.norm(Y_c_pos - t_pos, axis=1))
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
                X_c, Y_c_pos, _, _ = load_segment_data(base_dir, scene, task, demo, seg_check)
                min_len = min(X_c.shape[0], Y_c_pos.shape[0])
                X_c, Y_c_pos = X_c[:min_len], Y_c_pos[:min_len]
            except FileNotFoundError:
                break 
            
            segment_matches[seg_check] = []
            for obj_name, configs in target_positions.items():
                best_dist = float('inf')
                best_idx = -1
                best_t_pos = None
                best_t_ori = None
                
                for t_pos, t_ori in configs:
                    distances = np.linalg.norm(Y_c_pos - t_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    min_dist_in_segment = distances[closest_idx]
                    
                    if min_dist_in_segment < best_dist:
                        best_dist = min_dist_in_segment
                        best_idx = closest_idx
                        best_t_pos = t_pos
                        best_t_ori = t_ori
                
                if best_dist <= global_min_dist[obj_name] + 0.08 and best_dist < 0.25:
                    segment_matches[seg_check].append({
                        'obj_name': obj_name,
                        'time': X_c[best_idx, 0],
                        'target_pos': best_t_pos,
                        'target_ori': best_t_ori,
                        'action_type': manual_objects[obj_name]
                    })
                    print(f"   [FASE 1] -> Detectado '{obj_name}' en Seg {seg_check} (dist: {best_dist:.3f}m)")
            
            seg_check += 1

    # =========================================================================
    # SEGUNDA FASE: CONSTRUCCIÓN DE PROCESOS GAUSSIANOS POS+QUAT
    # =========================================================================
    segment_idx = 1
    while True:
        try:
            X, Y_pos, Y_quat, dt = load_segment_data(base_dir, scene, task, demo, segment_idx)
            min_len = min(X.shape[0], Y_pos.shape[0])
            X, Y_pos, Y_quat = X[:min_len], Y_pos[:min_len], Y_quat[:min_len]
            size = X.shape[0]
        except FileNotFoundError:
            break
            
        print(f">> Aprendiendo Segmento {segment_idx} (Pos + Ori) en silencio...")
        
        # --- FIX: Inyección Dinámica de Jitter ---
        Y_pos += np.random.normal(0, 1e-4, Y_pos.shape)
        
        Y_quat += np.random.normal(0, 1e-1, Y_quat.shape)
            
        Y_quat = Y_quat / np.linalg.norm(Y_quat, axis=1, keepdims=True)
        # -----------------------------------------

        target_t = X[-1, 0]
        orig_start_pos, orig_end_pos = Y_pos[0, :], Y_pos[-1, :]
        orig_start_quat, orig_end_quat = Y_quat[0, :], Y_quat[-1, :]
        
        via_t = np.array([X[0, 0], target_t]).reshape(-1, 1)
        vias_orig_pos = np.array([orig_start_pos, orig_end_pos])
        vias_orig_quat = np.array([orig_start_quat, orig_end_quat])
        
        curr_start_pos = orig_start_pos if previous_end_pos is None else previous_end_pos
        curr_end_pos = orig_end_pos
        previous_end_pos = curr_end_pos
        
        # 1. ENTRENAMOS GP PARA POSICIONES
        gp_mp = ProGpMp(X, Y_pos, via_t, vias_orig_pos, dim=3, demos=1, size=size, observation_noise=0.01)
        gp_mp.BlendedGpMp(gp_mp.ProGP)
        
        # 2. ENTRENAMOS GP PARA QUATERNIONES
        gp_quat = ProGpMp(X, Y_quat, via_t, vias_orig_quat, dim=4, demos=1, size=size, observation_noise=0.05)
        gp_quat.BlendedGpMp(gp_quat.ProGP)
        
        test_x = np.linspace(X[0, 0], X[-1, 0], size * 4).reshape(-1, 1)
        mean_gmp, var_gmp = gp_mp.predict_BlendedPos(test_x)
        mean_quat, var_quat = gp_quat.predict_BlendedPos(test_x)

        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        base_kernels = [cls(lengthscales=0.5) for cls in BaseClasses] 
        kernel = kernels.SeparateIndependent(base_kernels)
        mean_gmp_arr = np.vstack(mean_gmp) 
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

        BaseClassesQuat = (kernels.SquaredExponential,)*4
        base_kernelsQuat = [cls(lengthscales=0.5) for cls in BaseClassesQuat]
        kernelQuat = kernels.SeparateIndependent(base_kernelsQuat)
        mean_quat_arr = np.vstack(mean_quat)
        F_prior_full_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T, dtype=default_float()), axis=0)

        # Usamos tuplas para mapear Tiempo -> (Pos, Ori, Etiqueta)
        points_dict = {
            via_t[0, 0]: (curr_start_pos, orig_start_quat, "Start"),
            target_t: (curr_end_pos, orig_end_quat, "End")
        }

        if manual_objects and segment_idx in segment_matches:
            for match in segment_matches[segment_idx]:
                obj_name = match['obj_name']
                closest_time = match['time']
                target_pos = match['target_pos']
                target_ori = match['target_ori']
                action_type = match['action_type']
                
                virtual_name = obj_name
                if action_type == 'drop': 
                    virtual_name = f'drop_point_{obj_name}'
                    draw_debug_target(target_pos, f"DROP {obj_name}", is_grasp=False)
                elif action_type == 'grasp': 
                    draw_debug_target(target_pos, f"GRASP {obj_name}", is_grasp=True)
                
                if abs(closest_time - via_t[0, 0]) < 0.05:
                    points_dict[via_t[0, 0]] = (target_pos, target_ori, virtual_name)
                    print(f"   [VIA-POINT ASIGNADO] -> '{virtual_name}' fusionado en Start t={via_t[0, 0]:.4f}s")
                elif abs(closest_time - target_t) < 0.05:
                    points_dict[target_t] = (target_pos, target_ori, virtual_name)
                    print(f"   [VIA-POINT ASIGNADO] -> '{virtual_name}' fusionado en End t={target_t:.4f}s")
                else:
                    while closest_time in points_dict:
                        closest_time += 1e-4  
                    points_dict[closest_time] = (target_pos, target_ori, virtual_name)
                    print(f"   [VIA-POINT ASIGNADO] -> '{virtual_name}' fijado internamente en t={closest_time:.4f}s")

        points_info_list = [(t, val[0], val[1], val[2]) for t, val in points_dict.items()]
        points_info_list.sort(key=lambda x: x[0])

        via_t_cond = np.array([p[0] for p in points_info_list])
        vias_cond_pos = np.array([p[1] for p in points_info_list])
        vias_cond_quat = np.array([p[2] for p in points_info_list])
        obj_names_map = {idx: p[3] for idx, p in enumerate(points_info_list) if p[3] not in ["Start", "End"]}

        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 

        # 3. ACTUALIZACIÓN EXACTA POSICIONES
        Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(vias_cond_pos, dtype=default_float()), axis=0)
        F_obs_pos = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 
        dF_fn_pos = updates.exact(kernel, X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
        F_cond_pos = F_prior_full + dF_fn_pos(X_test)
        mean_corrected_pos = np.squeeze(F_cond_pos.numpy())

        # 4. ACTUALIZACIÓN EXACTA ORIENTACIONES
        Y_obs_quat = tf.expand_dims(tf.convert_to_tensor(vias_cond_quat, dtype=default_float()), axis=0)
        F_obs_quat = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T[obs_idx], dtype=default_float()), axis=0) 
        dF_fn_quat = updates.exact(kernelQuat, X_obs, Y_obs_quat, F_obs_quat, diag=0.0)
        F_cond_quat = F_prior_full_q + dF_fn_quat(X_test)
        mean_corrected_quat = np.squeeze(F_cond_quat.numpy())

        # Ensamblamos variables para el controlador PyBullet (Matriz 7D)
        mean_corrected_full = np.hstack([mean_corrected_pos, mean_corrected_quat])
        mean_prior_full = np.hstack([np.squeeze(mean_gmp_arr.T), np.squeeze(mean_quat_arr.T)])
        
        all_prior.append(mean_prior_full)
        all_corrected.append(mean_corrected_full)

        segments_info.append({
            'kernel': kernel, 'F_prior_full': F_prior_full, 
            'kernelQuat': kernelQuat, 'F_prior_full_q': F_prior_full_q,
            'via_t_cond': via_t_cond,
            'vias_cond_pos': vias_cond_pos, 'vias_cond_quat': vias_cond_quat, 
            'mean_gmp_arr': mean_gmp_arr, 'mean_quat_arr': mean_quat_arr, 
            'test_x': test_x,
            'obj_names': obj_names_map, 'mean_corrected': mean_corrected_full 
        })
        segment_idx += 1

    if segment_idx - 1 > 0:
        print("\n>> Cálculo GP finalizado. Abriendo Visor Interactivo y Controles en PyBullet...")
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
    CURRENT_SCENE = 1
    CURRENT_TASK = 1
    CURRENT_DEMO = 1

    # Define los objetos de la escena que vas a usar
    mis_objetos_manuales = {
        'banana': 'grasp', 
        'bucket_cap': 'grasp', 
        'bucket': 'drop'
    }
    
    chain_gp_pathwise_segments(
        scene=CURRENT_SCENE, 
        task=CURRENT_TASK, 
        demo=CURRENT_DEMO, 
        manual_objects=mis_objetos_manuales,
        robot_id=franka_robot,
        ee_idx=11 
    )
    
    p.disconnect()