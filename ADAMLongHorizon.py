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

# Tu librería local
from ProGP import ProGpMp

# ADAMSim imports
from scripts.adam import ADAM

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'cyan']

USE_SEGMENT_MEAN = True

# ============================================================================
# FUNCIONES DE ENTORNO DE PYBULLET (ESCENA PERSONALIZADA)
# ============================================================================

object_location_dict = {
    "table"  : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/wood_table/urdf/wood_table.urdf",
    "banana" : "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Objects/banana/banana.urdf",
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

def custom_adam_scene():
    """
    Escena minimalista: Una mesa justo delante de ADAM y dos plátanos encima.
    """
    object_dict = {
        "table": [[0.25, -0.7, -0.5], [-1, 0, 0, 1], True],
        
        # Plátanos sobre la mesa (Z=0.77 suele ser la altura de la mesa)
        "banana1": [[0.60, 0.15, 0.77], [-1, 0, 1, 1], False],
        "banana2": [[0.60, -0.15, 0.77], [-1, 0, 1, 1], False],
    }
    
    loaded_objects = {}
    for key in object_dict:
        urdf_name = "banana" if "banana" in key else key
        
        obj = p.loadURDF(
            object_location_dict[urdf_name], 
            object_dict[key][0], 
            object_dict[key][1], 
            useFixedBase=object_dict[key][2]
        )
        set_object_intrinsic_properties(obj, object_dict[key][2], key)
        loaded_objects[key] = obj
        
    return loaded_objects

def draw_debug_target(pos, label, is_grasp=True):
    color = [0, 1, 0] if is_grasp else [1, 0, 0] 
    length = 0.04
    p.addUserDebugLine([pos[0]-length, pos[1], pos[2]], [pos[0]+length, pos[1], pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1]-length, pos[2]], [pos[0], pos[1]+length, pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1], pos[2]-length], [pos[0], pos[1], pos[2]+length], color, 3)
    p.addUserDebugText(label, [pos[0], pos[1], pos[2]+0.05], textColorRGB=color, textSize=1.2)

# ============================================================================
# FUNCIONES DE AYUDA DE ADAM
# ============================================================================

def tareas_en_segundo_plano(adam):
    adam.hand_kinematics.move_hand_to_dofs('right', [1000, 1000, 1000, 1000, 1000, 1000])
    adam.hand_kinematics.move_hand_to_dofs('left', [1000, 1000, 1000, 1000, 1000, 1000])

# ============================================================================
# CLASE CONTROLADOR: VISUALIZADOR 3D INTERACTIVO PARA ADAM
# ============================================================================
class AdamGPController:
    def __init__(self, segments_info, all_prior, all_corrected_pos, all_corrected_quat, adam, loaded_objects):
        self.segments_info = segments_info
        self.all_prior = all_prior
        self.adam = adam
        self.loaded_objects = loaded_objects
        self.lines = []
        
        self.current_step = 0
        self.full_trajectory_pos = None
        self.full_trajectory_quat = None

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
        self.ax.set_title("Pathwise Conditioning Interactivo (ADAM)\n(Mueve los Sliders en la ventana de PyBullet)", fontsize=14)

        for i in range(len(segments_info)):
            self.ax.plot(all_prior[i][:, 0], all_prior[i][:, 1], all_prior[i][:, 2], '--', linewidth=2, c='grey', alpha=0.4)
            line, = self.ax.plot(segments_info[i]['mean_corrected_pos'][:, 0], 
                                 segments_info[i]['mean_corrected_pos'][:, 1], 
                                 segments_info[i]['mean_corrected_pos'][:, 2], '-', linewidth=3, c=colors[i % len(colors)])
            self.lines.append(line)

        point_colors = []
        for i in range(len(self.points)):
            seg_idx, v_idx = self.point_mapping[i]
            label = self.segments_info[seg_idx]['obj_names'].get(v_idx, "")
            
            if 'drop' in label.lower():
                point_colors.append('red')
            elif label in self.loaded_objects:
                point_colors.append('orange')
            elif 'grasp' in label.lower():
                point_colors.append('green')
            else:
                point_colors.append('blue')

        self.scatter = self.ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], s=150, c=point_colors, marker='D')

        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='orange', marker='D', linestyle='None', markersize=10, label='Objeto Real'))
        handles.append(mlines.Line2D([], [], color='green', marker='D', linestyle='None', markersize=10, label='Virtual Grasp Point'))
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
            init_quat = self.segments_info[seg_idx]['vias_cond_quat'][v_idx]
            
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

    def _build_full_trajectory(self):
        full_pos = []
        full_quat = []
        for info in self.segments_info:
            full_pos.append(info['mean_corrected_pos'])
            full_quat.append(info['mean_corrected_quat'])
            
        self.full_trajectory_pos = np.vstack(full_pos)
        self.full_trajectory_quat = np.vstack(full_quat)

    def recompute_segment(self, seg_idx):
        info = self.segments_info[seg_idx]
        
        X_obs = tf.convert_to_tensor(info['via_t_cond'].reshape(-1, 1), dtype=default_float())
        Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(info['vias_cond_pos'], dtype=default_float()), axis=0)

        obs_idx = [int(np.argmin(np.abs(info['test_x'].flatten() - t))) for t in info['via_t_cond']]
        F_obs_pos = tf.expand_dims(tf.convert_to_tensor(info['mean_gmp_arr'].T[obs_idx], dtype=default_float()), axis=0)

        dF_fn_pos = updates.exact(info['kernel'], X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
        X_test = tf.convert_to_tensor(info['test_x'], dtype=default_float())
        F_cond_pos = info['F_prior_full'] + dF_fn_pos(X_test)
        mean_corrected_pos = np.squeeze(F_cond_pos.numpy())
        
        self.lines[seg_idx].set_data_3d(mean_corrected_pos[:,0], mean_corrected_pos[:,1], mean_corrected_pos[:,2])
        self.segments_info[seg_idx]['mean_corrected_pos'] = mean_corrected_pos

        Y_obs_quat = tf.expand_dims(tf.convert_to_tensor(info['vias_cond_quat'], dtype=default_float()), axis=0)
        F_obs_quat = tf.expand_dims(tf.convert_to_tensor(info['mean_quat_arr'].T[obs_idx], dtype=default_float()), axis=0)
        
        dF_fn_quat = updates.exact(info['kernelQuat'], X_obs, Y_obs_quat, F_obs_quat, diag=0.0)
        F_cond_quat = info['F_prior_full_q'] + dF_fn_quat(X_test)
        mean_corrected_quat = np.squeeze(F_cond_quat.numpy())
        
        mean_corrected_quat = mean_corrected_quat / np.linalg.norm(mean_corrected_quat, axis=1, keepdims=True)
        self.segments_info[seg_idx]['mean_corrected_quat'] = mean_corrected_quat

    def step_simulation(self):
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
                
                if label in self.loaded_objects:
                    obj_id = self.loaded_objects[label]
                    _, ori = p.getBasePositionAndOrientation(obj_id)
                    offset = self.label_offsets.get(label, np.zeros(3))
                    p.resetBasePositionAndOrientation(obj_id, new_pos - offset, ori)
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
            self.fig.canvas.draw_idle()
            
        if self.full_trajectory_pos is not None and len(self.full_trajectory_pos) > 0:
            target_pos = self.full_trajectory_pos[self.current_step]
            target_quat = self.full_trajectory_quat[self.current_step]
            
            pose_objetivo = [target_pos.tolist(), target_quat.tolist()]
            
            self.adam.arm_kinematics.move_arm_to_pose_continuous(
                arm='left', target_pose=pose_objetivo, target_link='dummy', accurate=False
            )
            
            self.adam.step()
            
            self.current_step += 1
            if self.current_step >= len(self.full_trajectory_pos):
                self.current_step = 0

    def show(self):
        plt.show()

# ============================================================================
# LÓGICA DE DATOS Y PROCESOS GAUSSIANOS
# ============================================================================

def plot_segment_decomposition(segment_idx, X_main, Y_main, all_demos_data, test_x, mean_gmp, var_gmp, via_t, vias):
    font_size = 14
    fig = plt.figure(figsize=(18, 9), dpi=100)
    plt.subplots_adjust(left=0.05, right=0.98, wspace=0.6, hspace=0.8, bottom=0.1, top=0.95)
    
    # Gráfico 1: Descomposición Espacial 3D
    plt1 = plt.subplot2grid((11, 16), (0, 0), rowspan=11, colspan=7, projection='3d')
    plt1.scatter(vias[:, 0], vias[:, 1], vias[:, 2], s=400, c='green', marker='x', label='Vias (Start/End)')
    
    # Dibujamos la demo base
    plt1.scatter(Y_main[:, 0], Y_main[:, 1], Y_main[:, 2], s=25, c='darkgreen', marker='o', alpha=0.6, label='Base Demo')
    plt1.plot(mean_gmp[0], mean_gmp[1], mean_gmp[2], c='black', linewidth=4, label='GMP Mean 3D')
    
    plt1.legend(loc='upper left', frameon=False)
    plt1.set_xlabel('$x$/m'); plt1.set_ylabel('$y$/m'); plt1.set_zlabel('$z$/m')
    plt1.set_title(f'Segment {segment_idx}: 3D Spatial Path', fontsize=font_size+2)

    test_x_flat = test_x.flatten()

    # Gráficos de X, Y, Z vs Tiempo
    for i, color, label in zip(range(3), ['red', 'blue', 'purple'], ['$x_{GMP}$', '$y_{GMP}$', '$z_{GMP}$']):
        plt_dim = plt.subplot2grid((11, 16), (i*4, 8), rowspan=3, colspan=8)
        plt_dim.plot(test_x_flat, mean_gmp[i], c=color, linewidth=3, label=label)
        plt_dim.fill_between(test_x_flat, mean_gmp[i] - 2 * np.sqrt(np.abs(var_gmp[i])), 
                                          mean_gmp[i] + 2 * np.sqrt(np.abs(var_gmp[i])), color=color, alpha=0.3)
        plt_dim.scatter(via_t[:, 0], vias[:, i], s=400, c=color, marker='x')
        plt_dim.legend(loc='upper right', frameon=False)
        plt_dim.set_ylabel(f'Dim {i}/m')

    plt.show(block=True)

def load_segment_data(base_dir, segment_idx, gap=1):
    """
    Carga directamente los archivos segmentoX.txt de la carpeta base,
    eliminando puntos estáticos consecutivos y aplicando el gap.
    """
    file_name = f"segmento{segment_idx}.txt"
    file_path = os.path.join(base_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
            
    pos_data_full = np.loadtxt(file_path)
    
    # 1. ELIMINAR VALORES DUPLICADOS CONSECUTIVOS
    diffs = np.diff(pos_data_full, axis=0)
    mask = np.any(np.abs(diffs) > 1e-9, axis=1)
    mask = np.insert(mask, 0, True)
    pos_data_unique = pos_data_full[mask]
    
    # 2. APLICAR GAP
    pos_data = pos_data_unique[::gap, :]
    
    dt = 0.01 
    num_points = pos_data.shape[0]
    time_arr = np.arange(0, num_points * dt, dt).reshape(-1, 1)
    
    if pos_data.shape[1] >= 7:
        Y_pos = pos_data[:, :3]
        qw = pos_data[:, 3]
        qx = pos_data[:, 4]
        qy = pos_data[:, 5]
        qz = pos_data[:, 6]
        Y_quat = np.column_stack((qx, qy, qz, qw))
    else:
        Y_pos = pos_data[:, :3]
        default_q = p.getQuaternionFromEuler([math.pi, 0, 0])
        Y_quat = np.tile(default_q, (num_points, 1))

    return time_arr, Y_pos, Y_quat, dt

def chain_gp_pathwise_adam(adam, loaded_objects, manual_objects, base_dir):
    
    print(f"--- Procesando Aprendizaje y Pathwise Conditioning ---")

    previous_end_pos = None
    previous_end_quat = None
    
    all_corrected_pos = []
    all_corrected_quat = []
    
    segments_info = []
    all_prior = []

    # =========================================================================
    # NUEVO: PRE-CARGA DE TODOS LOS SEGMENTOS EN MEMORIA
    # =========================================================================
    print(">> Cargando todos los segmentos en memoria...")
    loaded_segments = {}
    seg_check = 1
    while True:
        try:
            X_c, Y_c, Y_q_c, dt_c = load_segment_data(base_dir, seg_check)
            min_len = min(X_c.shape[0], Y_c.shape[0])
            loaded_segments[seg_check] = {
                'X': X_c[:min_len],
                'Y_pos': Y_c[:min_len],
                'Y_quat': Y_q_c[:min_len],
                'dt': dt_c,
                'size': min_len
            }
            print(f"Segmento {seg_check} cargado y guardado en caché.")
            seg_check += 1
        except FileNotFoundError:
            break
            
    if not loaded_segments:
        print("[!] No se encontraron segmentos en la carpeta.")
        return

    # =========================================================================
    # FASE 0: OBTENCIÓN DINÁMICA DE COORDENADAS (APRENDIZAJE DE MÍNIMO GLOBAL)
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
                
                target_positions[obj_name] = []
                puntos_base = [[center_x, center_y, z_top]]
                
                for pt in puntos_base:
                    t_pos = np.array(pt)
                    if action_type == 'drop': t_pos[2] += 0.1
                    elif action_type == 'grasp': t_pos[2] -= 0.02
                    target_positions[obj_name].append(t_pos)
                    
                global_min_dist[obj_name] = float('inf')
        
        # Iteramos usando los segmentos ya cargados en memoria
        for s_idx, data in loaded_segments.items():
            Y_c = data['Y_pos']
            for obj_name, t_pos_list in target_positions.items():
                for t_pos in t_pos_list:
                    min_d = np.min(np.linalg.norm(Y_c - t_pos, axis=1))
                    if min_d < global_min_dist[obj_name]:
                        global_min_dist[obj_name] = min_d

    # =========================================================================
    # FASE 1: ASIGNACIÓN SEGMENTO A SEGMENTO BASADA EN LO APRENDIDO
    # =========================================================================
    segment_matches = {}
    if manual_objects:
        for s_idx, data in loaded_segments.items():
            segment_matches[s_idx] = []
            X_c = data['X']
            Y_c = data['Y_pos']
            Y_q_c = data['Y_quat']
            
            for obj_name, t_pos_list in target_positions.items():
                
                best_dist = float('inf')
                best_idx = -1
                best_t_pos = None
                
                for t_pos in t_pos_list:
                    distances = np.linalg.norm(Y_c - t_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    min_dist_in_segment = distances[closest_idx]
                    
                    if min_dist_in_segment < best_dist:
                        best_dist = min_dist_in_segment
                        best_idx = closest_idx
                        best_t_pos = t_pos
                
                if best_dist <= global_min_dist[obj_name] + 0.08 and best_dist < 0.25:
                    action_type = manual_objects[obj_name]
                    segment_matches[s_idx].append({
                        'obj_name': obj_name,
                        'time': X_c[best_idx, 0],
                        'target_pos': best_t_pos,
                        'target_quat': Y_q_c[best_idx, :].copy(),
                        'action_type': action_type
                    })
                    print(f"   [FASE 1] -> Detectado '{obj_name}' en Seg {s_idx} (dist: {best_dist:.3f}m)")

    # =========================================================================
    # SEGUNDA FASE: CONSTRUCCIÓN DE PROCESOS GAUSSIANOS CON ASIGNACIÓN DIRECTA
    # =========================================================================
    for s_idx, data in loaded_segments.items():
        print(f">> Aprendiendo Segmento {s_idx} ...")
        
        X = data['X']
        Y_pos = data['Y_pos']
        Y_quat = data['Y_quat']
        size = data['size']

        target_t = X[-1, 0]
        orig_start_pos, orig_end_pos = Y_pos[0, :], Y_pos[-1, :]
        orig_start_quat, orig_end_quat = Y_quat[0, :], Y_quat[-1, :]
        
        via_t = np.array([X[0, 0], target_t]).reshape(-1, 1)
        vias_orig_pos = np.array([orig_start_pos, orig_end_pos])
        vias_orig_quat = np.array([orig_start_quat, orig_end_quat])
        
        curr_start_pos = orig_start_pos if previous_end_pos is None else previous_end_pos
        curr_end_pos = orig_end_pos
        previous_end_pos = curr_end_pos
        
        curr_start_quat = orig_start_quat if previous_end_quat is None else previous_end_quat
        curr_end_quat = orig_end_quat
        previous_end_quat = curr_end_quat
        
        test_x = np.linspace(X[0, 0], X[-1, 0], size * 4).reshape(-1, 1)
        
        if not USE_SEGMENT_MEAN:
            # 1. ENTRENAMOS EL GP DE POSICIÓN
            gp_mp = ProGpMp(X, Y_pos, via_t, vias_orig_pos, dim=3, demos=1, size=size, observation_noise=0.1)
            gp_mp.BlendedGpMp(gp_mp.ProGP)
            mean_gmp, var_gmp = gp_mp.predict_BlendedPos(test_x)
            plot_segment_decomposition(
                s_idx, X, Y_pos, [(X, Y_pos)], test_x, 
                mean_gmp, var_gmp, via_t, vias_orig_pos
            )
            
            # 2. ENTRENAMOS EL GP DE ORIENTACIÓN (QUATERNIONES)
            gp_quat = ProGpMp(X, Y_quat, via_t, vias_orig_quat, dim=4, demos=1, size=size, observation_noise=0.03)
            gp_quat.BlendedGpMp(gp_quat.ProGP)
            
            mean_quat, _ = gp_quat.predict_BlendedPos(test_x)
        

        if USE_SEGMENT_MEAN:
            mean_pos_data = np.mean(Y_pos, axis=0)
            mean_gmp_arr = np.tile(mean_pos_data, (test_x.shape[0], 1)).T
            
            mean_quat_data = np.mean(Y_quat, axis=0)
            mean_quat_data = mean_quat_data / np.linalg.norm(mean_quat_data)
            mean_quat_arr = np.tile(mean_quat_data, (test_x.shape[0], 1)).T
        else:
            mean_gmp_arr = np.vstack(mean_gmp)
            mean_quat_arr = np.vstack(mean_quat) 
            
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)
        F_prior_full_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T, dtype=default_float()), axis=0)
        
        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        base_kernels = [cls(lengthscales=0.5) for cls in BaseClasses] 
        kernel = kernels.SeparateIndependent(base_kernels)

        BaseClassesQuat = (kernels.SquaredExponential,)*4
        base_kernelsQuat = [cls(lengthscales=0.5) for cls in BaseClassesQuat]
        kernelQuat = kernels.SeparateIndependent(base_kernelsQuat)

        points_dict = {
            via_t[0, 0]: (curr_start_pos, curr_start_quat, "Start"),
            target_t: (curr_end_pos, curr_end_quat, "End")
        }

        if manual_objects and s_idx in segment_matches:
            for match in segment_matches[s_idx]:
                obj_name = match['obj_name']
                closest_time = match['time']
                target_pos = match['target_pos']
                target_quat = match['target_quat']
                action_type = match['action_type']
                
                virtual_name = obj_name
                if action_type == 'drop': 
                    virtual_name = f'drop_point_{obj_name}'
                    draw_debug_target(target_pos, f"DROP {obj_name}", is_grasp=False)
                elif action_type == 'grasp': 
                    draw_debug_target(target_pos, f"GRASP {obj_name}", is_grasp=True)
                
                if abs(closest_time - via_t[0, 0]) <= 0.005:
                    points_dict[via_t[0, 0]] = (target_pos, target_quat, virtual_name)
                    print(f"   [VIA-POINT FUSIONADO] -> '{virtual_name}' fusionado en Start t={via_t[0, 0]:.4f}s")
                elif abs(closest_time - target_t) <= 0.005:
                    points_dict[target_t] = (target_pos, target_quat, virtual_name)
                    print(f"   [VIA-POINT FUSIONADO] -> '{virtual_name}' fusionado en End t={target_t:.4f}s")
                else:
                    while closest_time in points_dict:
                        closest_time += 1e-4  
                    points_dict[closest_time] = (target_pos, target_quat, virtual_name)
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

        # APLICAMOS PATHWISE CONDITIONING A POSICIÓN
        Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(vias_cond_pos, dtype=default_float()), axis=0)
        F_obs_pos = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 
        dF_fn_pos = updates.exact(kernel, X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
        F_cond_pos = F_prior_full + dF_fn_pos(X_test)
        mean_corrected_pos = np.squeeze(F_cond_pos.numpy())

        # APLICAMOS PATHWISE CONDITIONING A ORIENTACIÓN
        Y_obs_quat = tf.expand_dims(tf.convert_to_tensor(vias_cond_quat, dtype=default_float()), axis=0)
        F_obs_quat = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T[obs_idx], dtype=default_float()), axis=0) 
        dF_fn_quat = updates.exact(kernelQuat, X_obs, Y_obs_quat, F_obs_quat, diag=0.0)
        F_cond_quat = F_prior_full_q + dF_fn_quat(X_test)
        mean_corrected_quat = np.squeeze(F_cond_quat.numpy())
        mean_corrected_quat = mean_corrected_quat / np.linalg.norm(mean_corrected_quat, axis=1, keepdims=True)

        all_corrected_pos.append(mean_corrected_pos)
        all_corrected_quat.append(mean_corrected_quat)
        
        mean_prior = np.squeeze(mean_gmp_arr.T)
        all_prior.append(mean_prior)
        
        segments_info.append({
            'kernel': kernel, 'F_prior_full': F_prior_full, 
            'kernelQuat': kernelQuat, 'F_prior_full_q': F_prior_full_q,
            'via_t_cond': via_t_cond,
            'vias_cond_pos': vias_cond_pos, 'vias_cond_quat': vias_cond_quat, 
            'mean_gmp_arr': mean_gmp_arr, 'mean_quat_arr': mean_quat_arr, 
            'test_x': test_x,
            'obj_names': obj_names_map, 
            'mean_corrected_pos': mean_corrected_pos,
            'mean_corrected_quat': mean_corrected_quat
        })

    if len(all_corrected_pos) == 0:
        print("[!] No se encontraron segmentos. Revisa la ruta de 'SementsFolder'.")
        return

    full_trajectory_pos = np.vstack(all_corrected_pos)
    full_trajectory_quat = np.vstack(all_corrected_quat)

    # =========================================================================
    # FASE DE EJECUCIÓN EN ADAM
    # =========================================================================
    print("\n>> Cálculo finalizado. Moviendo a ADAM al punto de inicio...")
    
    pose_inicial = [full_trajectory_pos[0].tolist(), full_trajectory_quat[0].tolist()]
    
    adam.arm_kinematics.move_arm_to_pose_continuous(
        arm='left', target_pose=pose_inicial, target_link='dummy', accurate=False
    )
    adam.step()
        
    print(">> ¡Posición inicial alcanzada! Esperando 1 segundo...")
    adam.wait(1.0)

    print("\n>> Abriendo Visor Interactivo (Matplotlib + Sliders) ...")
    plotter = AdamGPController(segments_info, all_prior, all_corrected_pos, all_corrected_quat, adam, loaded_objects)
    plotter.show()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    robot_urdf_path = os.path.join(base_path, "AdamSim", "models", "robot", "rb1_base_description", "robots", "robotDummy.urdf")
    
    print('Cargando simulador ADAM...')
    adam = ADAM(robot_urdf_path, useRealTimeSimulation=True, used_fixed_base=True, use_ros=False)
    print('¡ADAM Listo!')
    
    initial_left_pose = [-5.688085381184713, 5.571132170944967, 0.8321369330035608 , -1.486776666050293, -0.6288579146014612, -0.5840128103839319]
    
    adam.arm_kinematics.move_arm_joints_to_angles(arm='left', angles=initial_left_pose)
    adam.step()
    
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 1.0])

    # Buscamos los plátanos en la escena para hacer de via-points de agarre
    mis_objetos_manuales = {
        'banana1': 'grasp', 
        'banana2': 'grasp'
    }

    # Cargamos la escena personalizada con la mesa y los plátanos
    loaded_objects = custom_adam_scene()

    # Ejecutamos el pipeline completo
    chain_gp_pathwise_adam(
        adam=adam, 
        loaded_objects=loaded_objects,
        manual_objects=mis_objetos_manuales,
        base_dir='/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Segmentation-and-Grouping-Model/SegmentationProb/scripts/SegmentsFolder'
    )
    
    p.disconnect()