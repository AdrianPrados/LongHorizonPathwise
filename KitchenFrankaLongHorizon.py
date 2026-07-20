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

# ============================================================================
# FUNCIONES DE AYUDA (VISUALIZACIÓN Y DATOS)
# ============================================================================

def draw_debug_target(pos, label, is_grasp=True):
    color = [0, 1, 0] if is_grasp else [1, 0, 0] 
    length = 0.04
    p.addUserDebugLine([pos[0]-length, pos[1], pos[2]], [pos[0]+length, pos[1], pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1]-length, pos[2]], [pos[0], pos[1]+length, pos[2]], color, 3)
    p.addUserDebugLine([pos[0], pos[1], pos[2]-length], [pos[0], pos[1], pos[2]+length], color, 3)
    p.addUserDebugText(label, [pos[0], pos[1], pos[2]+0.05], textColorRGB=color, textSize=1.2)

def load_segment_data(base_dir, task_name, demo_idx, segment_idx,gap=20):
    """
    Lee los archivos .txt extraídos y normalizados del Franka Kitchen.
    Contienen: x, y, z, qx, qy, qz, qw
    """
    folder_path = os.path.join(base_dir, task_name)
    file_name = f"demo_{demo_idx}_segmento_{segment_idx}.txt" 
    file_path = os.path.join(folder_path, file_name)
    print(file_path)
    
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
    data_full = np.loadtxt(file_path)
    data = data_full[::gap, :]
    dt = 0.01 
    num_points = data.shape[0]
    #time_arr = np.arange(0, num_points * dt, dt).reshape(-1, 1)
    time_arr = (np.arange(num_points) * dt).reshape(-1, 1)
    
    Y_pos = data[:, 0:3]
    Y_quat = data[:, 3:7]
    
    return time_arr, Y_pos, Y_quat, dt

# ============================================================================
# CLASE CONTROLADOR: VISUALIZADOR 3D INTERACTIVO PARA FRANKA KITCHEN
# ============================================================================
class KitchenGPController:
    def __init__(self, segments_info, all_prior, all_corrected_pos, all_corrected_quat, env, robot, ee_idx):
        self.segments_info = segments_info
        self.all_prior = all_prior
        self.env = env
        self.robot = robot
        self.ee_idx = ee_idx
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
        
        # Configurar Matplotlib
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)
        self.ax.set_title("Pathwise Conditioning - Franka Kitchen", fontsize=14)

        for i in range(len(segments_info)):
            self.ax.plot(all_prior[i][:, 0], all_prior[i][:, 1], all_prior[i][:, 2], '--', linewidth=2, c='grey', alpha=0.4)
            line, = self.ax.plot(segments_info[i]['mean_corrected_pos'][:, 0], 
                                 segments_info[i]['mean_corrected_pos'][:, 1], 
                                 segments_info[i]['mean_corrected_pos'][:, 2], '-', linewidth=3, c=colors[i % len(colors)])
            self.lines.append(line)

        self.scatter = self.ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], s=150, c='blue', marker='D')

        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Via-points'))
        self.ax.legend(handles=handles, fontsize=10, loc='best')

        self._build_full_trajectory()
        
        # Iniciar temporizador
        self.timer = self.fig.canvas.new_timer(interval=30)
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

    def step_simulation(self):
        if self.full_trajectory_pos is not None and len(self.full_trajectory_pos) > 0:
            target_pos = self.full_trajectory_pos[self.current_step]
            target_quat = self.full_trajectory_quat[self.current_step]
            
            # Cinemática inversa para el brazo Franka
            joint_poses = p.calculateInverseKinematics(
                self.robot.id, self.ee_idx, 
                targetPosition=target_pos, 
                targetOrientation=target_quat, 
                maxNumIterations=100
            )
            
            # Aplicar control de posición
            movable_joints = [j for j in range(p.getNumJoints(self.robot.id)) if p.getJointInfo(self.robot.id, j)[2] in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]]
            
            for i, joint_idx in enumerate(movable_joints):
                if i < 7: # Solo los 7 DoF del brazo
                    p.setJointMotorControl2(
                        bodyUniqueId=self.robot.id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses[i],
                        force=500,
                        maxVelocity=5.0
                    )
            
            self.env.update()
            
            self.current_step += 1
            if self.current_step >= len(self.full_trajectory_pos):
                self.current_step = 0

    def show(self):
        plt.show()

# ============================================================================
# LÓGICA PRINCIPAL: GMP + PATHWISE CONDITIONING
# ============================================================================
def chain_gp_pathwise_kitchen(env, robot, manual_objects, base_dir, task_name, demo_idx, ee_idx=8):
    
    print(f"--- Procesando Pathwise Conditioning para {task_name} | Demo {demo_idx} ---")

    all_corrected_pos, all_corrected_quat, all_prior, segments_info = [], [], [], []
    previous_end_pos, previous_end_quat = None, None

    # =========================================================================
    # FASE 0: PRE-CARGA Y CÁLCULO DE COORDENADAS DE OBJETOS
    # =========================================================================
    target_positions = {}
    global_min_dist = {}
    
    if manual_objects:
        for obj_name, obj_data in manual_objects.items():
            body_id = obj_data['id']
            link_idx = obj_data['link']
            action_type = obj_data['action']
            
            if link_idx == -1:
                pos, _ = p.getBasePositionAndOrientation(body_id)
            else:
                pos = p.getLinkState(body_id, link_idx)[0]
                
            t_pos = np.array(pos)
            if action_type == 'drop': t_pos[2] += 0.1
            elif action_type == 'grasp': t_pos[2] -= 0.02
            
            target_positions[obj_name] = [t_pos]
            global_min_dist[obj_name] = float('inf')

        # Buscar la distancia mínima global analizando todos los segmentos
        seg_check = 1
        while True:
            try:
                X_c, Y_c, _, _ = load_segment_data(base_dir, task_name, demo_idx, seg_check)
                for obj_name, t_pos_list in target_positions.items():
                    for t_pos in t_pos_list:
                        min_d = np.min(np.linalg.norm(Y_c - t_pos, axis=1))
                        if min_d < global_min_dist[obj_name]:
                            global_min_dist[obj_name] = min_d
            except FileNotFoundError:
                break
            seg_check += 1

    # =========================================================================
    # FASE 1: ASIGNACIÓN DE VÍA-POINTS A SEGMENTOS
    # =========================================================================
    segment_matches = {}
    if manual_objects:
        seg_check = 1
        while True:
            try:
                X_c, Y_c, Y_q_c, _ = load_segment_data(base_dir, task_name, demo_idx, seg_check)
            except FileNotFoundError:
                break 
            
            segment_matches[seg_check] = []
            for obj_name, t_pos_list in target_positions.items():
                best_dist, best_idx, best_t_pos = float('inf'), -1, None
                for t_pos in t_pos_list:
                    distances = np.linalg.norm(Y_c - t_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < best_dist:
                        best_dist, best_idx, best_t_pos = distances[closest_idx], closest_idx, t_pos
                
                # Tolerancia de acercamiento
                if best_dist <= global_min_dist[obj_name] + 0.08 and best_dist < 0.25:
                    segment_matches[seg_check].append({
                        'obj_name': obj_name,
                        'time': X_c[best_idx, 0],
                        'target_pos': best_t_pos,
                        'target_quat': Y_q_c[best_idx, :].copy(),
                        'action_type': manual_objects[obj_name]['action']
                    })
                    print(f"   -> Objeto '{obj_name}' detectado en Segmento {seg_check} (dist: {best_dist:.3f}m)")
            
            seg_check += 1

    # =========================================================================
    # FASE 2: APRENDIZAJE GMP Y CONDITIONING
    # =========================================================================
    segment_idx = 1
    while True:
        try:
            X, Y_pos, Y_quat, dt = load_segment_data(base_dir, task_name, demo_idx, segment_idx)
            size = X.shape[0]
        except FileNotFoundError:
            break
            
        print(f">> Aplicando GMP y Pathwise al Segmento {segment_idx}...")
        
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
        
        # ENTRENAR GP POSICIÓN
        gp_pos = ProGpMp(X, Y_pos, via_t, vias_orig_pos, dim=3, demos=1, size=size, observation_noise=0.01)
        gp_pos.BlendedGpMp(gp_pos.ProGP)
        test_x = np.linspace(X[0, 0], X[-1, 0], size * 4).reshape(-1, 1)
        mean_gmp_pos, _ = gp_pos.predict_BlendedPos(test_x)
        
        # ENTRENAR GP ORIENTACIÓN
        gp_quat = ProGpMp(X, Y_quat, via_t, vias_orig_quat, dim=4, demos=1, size=size, observation_noise=0.01)
        gp_quat.BlendedGpMp(gp_quat.ProGP)
        mean_gmp_quat, _ = gp_quat.predict_BlendedPos(test_x)

        # CONFIGURACIÓN KERNELS
        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        kernel = kernels.SeparateIndependent([cls(lengthscales=0.05) for cls in BaseClasses])
        mean_gmp_arr = np.vstack(mean_gmp_pos) 
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

        BaseClassesQuat = (kernels.SquaredExponential,)*4
        kernelQuat = kernels.SeparateIndependent([cls(lengthscales=0.05) for cls in BaseClassesQuat])
        mean_quat_arr = np.vstack(mean_gmp_quat)
        F_prior_full_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T, dtype=default_float()), axis=0)

        # INSERTAR PUNTOS CONDICIONALES
        points_dict = {
            via_t[0, 0]: (curr_start_pos, curr_start_quat, "Start"),
            target_t: (curr_end_pos, curr_end_quat, "End")
        }

        if manual_objects and segment_idx in segment_matches:
            for match in segment_matches[segment_idx]:
                closest_time = match['time']
                target_pos = match['target_pos']
                target_quat = match['target_quat']
                virtual_name = match['obj_name']
                
                if match['action_type'] == 'drop': draw_debug_target(target_pos, f"DROP {virtual_name}", is_grasp=False)
                else: draw_debug_target(target_pos, f"GRASP {virtual_name}", is_grasp=True)
                
                if abs(closest_time - via_t[0, 0]) <= 0.005: points_dict[via_t[0, 0]] = (target_pos, target_quat, virtual_name)
                elif abs(closest_time - target_t) <= 0.005: points_dict[target_t] = (target_pos, target_quat, virtual_name)
                else:
                    while closest_time in points_dict: closest_time += 1e-4  
                    points_dict[closest_time] = (target_pos, target_quat, virtual_name)

        points_info_list = sorted([(t, val[0], val[1], val[2]) for t, val in points_dict.items()], key=lambda x: x[0])

        via_t_cond = np.array([p[0] for p in points_info_list])
        vias_cond_pos = np.array([p[1] for p in points_info_list])
        vias_cond_quat = np.array([p[2] for p in points_info_list])
        obj_names_map = {idx: p[3] for idx, p in enumerate(points_info_list) if p[3] not in ["Start", "End"]}

        # PATHWISE UPDATES (Posición)
        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        Y_obs_pos = tf.expand_dims(tf.convert_to_tensor(vias_cond_pos, dtype=default_float()), axis=0)
        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        F_obs_pos = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 
        
        dF_fn_pos = updates.exact(kernel, X_obs, Y_obs_pos, F_obs_pos, diag=0.0)
        X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 
        mean_corrected_pos = np.squeeze((F_prior_full + dF_fn_pos(X_test)).numpy())

        # PATHWISE UPDATES (Orientación)
        Y_obs_quat = tf.expand_dims(tf.convert_to_tensor(vias_cond_quat, dtype=default_float()), axis=0)
        F_obs_quat = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T[obs_idx], dtype=default_float()), axis=0) 
        
        dF_fn_quat = updates.exact(kernelQuat, X_obs, Y_obs_quat, F_obs_quat, diag=0.0)
        mean_corrected_quat = np.squeeze((F_prior_full_q + dF_fn_quat(X_test)).numpy())
        mean_corrected_quat = mean_corrected_quat / np.linalg.norm(mean_corrected_quat, axis=1, keepdims=True)

        all_corrected_pos.append(mean_corrected_pos)
        all_corrected_quat.append(mean_corrected_quat)
        all_prior.append(np.squeeze(mean_gmp_arr.T))

        segments_info.append({
            'kernel': kernel, 'F_prior_full': F_prior_full, 
            'via_t_cond': via_t_cond, 'vias_cond_pos': vias_cond_pos, 'vias_cond_quat': vias_cond_quat, 
            'mean_gmp_arr': mean_gmp_arr, 'test_x': test_x, 'obj_names': obj_names_map, 
            'mean_corrected_pos': mean_corrected_pos, 'mean_corrected_quat': mean_corrected_quat
        })
        segment_idx += 1

    if len(all_corrected_pos) > 0:
        print("\n>> Abriendo Visor Interactivo y reproduciendo en PyBullet...")
        plotter = KitchenGPController(segments_info, all_prior, all_corrected_pos, all_corrected_quat, env, robot, ee_idx)
        plotter.show()
    else:
        print("[!] No se encontraron segmentos para procesar.")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    
    # 1. Iniciar PyBullet y Cargar Entorno
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    print("Cargando el entorno de Franka Kitchen...")
    env = Environment()
    env.load()
    robot = Robot()
    
    # Buscamos el End-Effector (panda_link8)
    ee_link_index = -1
    for j in range(p.getNumJoints(robot.id)):
        if p.getJointInfo(robot.id, j)[12].decode('utf-8') == 'panda_link8':
            ee_link_index = j
            break
    if ee_link_index == -1: ee_link_index = 8 # Valor por defecto seguro

    # 2. Configuración de la Tarea y Objetos
    # Aquí puedes añadir los objetos por los que quieres que pase la trayectoria
    # Si 'link' es -1, buscará la base del objeto (como el kettle)
    # Si 'link' es >0, buscará esa pieza exacta dentro de la cocina (ej. Microwave = 54)
    
    OBJETOS_A_CONDICIONAR = {
        'microwave': {'action': 'grasp', 'id': env.kitchen_id, 'link': 54},
        'kettle': {'action': 'grasp', 'id': env.kettle_id, 'link': -1},
        'light_switch': {'action': 'grasp', 'id': env.kitchen_id, 'link': 28}
    }
    
    # Carpetas de datos extraídos
    CARPETA_SEGMENTOS = "/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Segmentation-and-Grouping-Model/SegmentationProb/scripts/KitchenFranka_SegmentsFolder"
    TAREA_A_PROCESAR = "traj_friday_kettle_bottomknob_hinge_slide" 
    DEMO_SELECCIONADA = 1

    # 3. Lanzar Algoritmo
    chain_gp_pathwise_kitchen(
        env=env,
        robot=robot,
        manual_objects=OBJETOS_A_CONDICIONAR,
        base_dir=CARPETA_SEGMENTOS,
        task_name=TAREA_A_PROCESAR,
        demo_idx=DEMO_SELECCIONADA,
        ee_idx=ee_link_index
    )
    
    p.disconnect()