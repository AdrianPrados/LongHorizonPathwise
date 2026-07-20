import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN

# Intenta importar las librerías locales originales si existen
try:
    from utils import *
    import drawData2D as scr2
    import downsampling as ds
except ImportError:
    pass

import time as t
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks, savgol_filter
import os
import pickle
import time

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k'  ]

def normal(x, mean=0, stdev=1):
    return (1 / (stdev * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / stdev)**2)

def calc_time_deriv(time, data):
    n_pts, n_dims = np.shape(data)
    deriv = np.zeros((n_pts-1, n_dims))
    for i in range(n_pts-1):
        deriv[i] = (data[i + 1] - data[i]) / (time[i + 1] - time[i])
    return deriv

def calc_jerk_in_time(time, position):
    data = position
    for _ in range(3):
        data = calc_time_deriv(time, data)
    return data

def calc_acceleration(time, position):
    data = position
    for _ in range(2):
        data = calc_time_deriv(time, data)
    return data

def moving_average(data, window_size):
    avg_data = []
    for i in range(len(data) - window_size):
        avg_data.append(np.mean(data[i:i+window_size]))
    return np.array(avg_data)

def detect_brutal_changes(data, change_threshold, segment_size, acceleration):
    jerk_diff = np.abs(np.diff(data))
    segments = []
    start_index = 0
    data_prev = acceleration[0]
    for i in range (len(data)-1):
        if jerk_diff[i] - start_index >= 10e-2 or (np.sign(acceleration[i]) != np.sign(data_prev)):
            segments.append(i)
        start_index = jerk_diff[i]
        data_prev = acceleration[i]
    if len(segments) == 0:
        segments.append(0)
    else:
        segments.append(i+1)
    return segments

def count_thresh(data, threshold, segment_size, grace_threshold, acceleration):
    segments = [0]
    count = 0
    grace_count = 0
    grace_threshold = 10
    data_prev = acceleration[0]
    for i in range(len(data)):
        if data[i] >= threshold and (np.sign(acceleration[i]) != np.sign(data_prev)):
            segments.append(i)
        else:
            if grace_count > grace_threshold:
                count = 0
            else:
                grace_count = grace_count + 1
                count = count + 1
        data_prev = acceleration[i]
    return segments

def normalize_time_series(time, data, target_length=100):
    new_time = np.linspace(time[0], time[-1], target_length)
    new_data = np.zeros((target_length, data.shape[1]))
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(new_time, time, data[:, i])
    return new_time, new_data

def signed_norm(vector):
    norm = np.linalg.norm(vector, axis=1)  
    sign = np.sign(vector[:, 0])  
    return norm * sign 

def segment(time, data, base_thresh=1000, segment_size=10, window_size=64, grace_thresh=32, plot=False, mode="variations"):
    if len(time) != len(data):
        raise ValueError("El tamaño del array de tiempo no coincide con el tamaño del array de datos.")
    if np.isnan(time).any() or np.isnan(data).any():
        raise ValueError("Existen valores NaN en los datos o en el tiempo.")
    if base_thresh < 0 or segment_size <= 0 or window_size <= 0 or grace_thresh < 0:
        raise ValueError("Los valores de los parámetros no son válidos.")
    
    jerk = calc_jerk_in_time(time, data)
    total_jerk = signed_norm(jerk)
    avg_jerk = moving_average(total_jerk, window_size)
    norm_avg_jerk = avg_jerk / np.max(avg_jerk)
    
    acceleration = calc_acceleration(time, data)
    total_acceleration = signed_norm(acceleration)
    avg_acceleration = moving_average(total_acceleration, window_size)
    norm_avg_acceleration = avg_acceleration / np.max(avg_acceleration)
    
    if mode == "variations":
        segments = detect_brutal_changes(norm_avg_jerk, base_thresh, segment_size, norm_avg_acceleration)
    elif mode == "threshold":
        segments = count_thresh(norm_avg_jerk, base_thresh, segment_size, grace_thresh, norm_avg_acceleration)
    else:
        raise ValueError("Invalid Mode")
        
    for i in range(1, len(segments)):
        segments[i] = int(segments[i] + window_size // 2)
        
    return segments

def calc_segment_prob(segment_list, data_len, window_size, plot=True):
    probabilities = np.ones((data_len,))
    for segment_point in segment_list:
        probabilities = probabilities + normal(np.linspace(0, 1, data_len), mean=segment_point / data_len, stdev=window_size / data_len)
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot=False):
    probabilities = np.ones((data_len,))
    for segment_list in list_of_list_of_segments:
        segment_probabilities = calc_segment_prob(segment_list[1:], data_len, window_size, plot=False)
        probabilities = probabilities * segment_probabilities
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def probabilistically_combine(list_of_list_of_segments, data_len, window_size, n_samples=10, n_pass=2, plot=False):
    probabilities = calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot)
    peaks, _ = find_peaks(probabilities)
    
    keypoints = peaks
    sorted_keys = np.sort(keypoints)
    sorted_keys = np.insert(sorted_keys, 0, 0)
    sorted_keys = np.append(sorted_keys, data_len)
    
    final_keys = []
    cur_key = 0
    cur_key_group = []
    local_window_size = 200 
    
    for i in range(len(sorted_keys)):
        if sorted_keys[i] <= cur_key + local_window_size:
            cur_key_group.append(sorted_keys[i])
        else:
            if len(cur_key_group) >= n_pass:
                final_keys.append(int(np.mean(cur_key_group)))
            cur_key = sorted_keys[i]
            cur_key_group = [cur_key]
            
    if len(cur_key_group) >= n_pass:
        final_keys.append(int(np.mean(cur_key_group)))
        
    keypoints = np.insert(final_keys, 0, int(0))
    keypoints = np.append(keypoints, int(data_len))
    
    if len(keypoints) > 1 and abs(keypoints[0] - keypoints[1]) < local_window_size:
        keypoints = np.delete(keypoints, 1)
    if len(keypoints) > 1 and abs(keypoints[-1] - keypoints[-2]) < local_window_size:
        keypoints = np.delete(keypoints, -2)
    
    return keypoints

def full_segmentation(time, list_of_data, base_thresh=1000, segment_size=256, window_size=64, grace_thresh=32, n_samples=10, n_pass=2, plot=False):
    list_of_segments = [segment(time, data, base_thresh=base_thresh, segment_size=segment_size, window_size=window_size, grace_thresh=grace_thresh, plot=plot) for data in list_of_data]
    segments = probabilistically_combine(list_of_segments, len(time), window_size, n_samples=n_samples, n_pass=n_pass, plot=plot)
    return segments

# =====================================================================
# FUNCIONES ESPECÍFICAS PARA FRANKA LHT SIMULATOR
# =====================================================================
def read_lht_trajectory(csv_path):
    data = np.loadtxt(csv_path, delimiter=',')
    time = data[:, 0]
    joint_pos = data[:, 1:8]
    traj_pos = data[:, 8:11]
    gripper_pos = data[:, 12:13] 
    wrench_frc = np.zeros((len(time), 6))
    return time, joint_pos, traj_pos, wrench_frc, gripper_pos

def load_lht_context(pkl_path):
    with open(pkl_path, 'rb') as f:
        location_dict = pickle.load(f)
    return location_dict

def main_lht(csv_path):
    time, joint_pos, traj_pos, wrench_frc, gripper_pos = read_lht_trajectory(csv_path)
    
    thresh = 0.25 
    ssize = 32
    wsize = 64
    gthresh = 32
    
    traj_segments = segment(time, traj_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False, mode="variations")
    gripper_segments = segment(time, gripper_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False, mode="variations")
    
    segments_to_combine = [traj_segments, gripper_segments]
    return segments_to_combine, traj_pos

def main3d(i):
    pass

def main2d(i):
    pass

if __name__ == '__main__':
    all_segments = []
    all_demos = []
    segment_data = []
    
    # -------------------------------------------------------------
    # MODO DE EJECUCIÓN (LHT procesará la estructura de carpetas)
    mode = "LHT"  
    # -------------------------------------------------------------

    if mode == "LHT":
        LHT_BASE_DIR = '/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Trajectory' 
        OUTPUT_BASE_DIR = 'LHT_SegmentsFolder_pruebas'

        if not os.path.exists(OUTPUT_BASE_DIR):
            os.makedirs(OUTPUT_BASE_DIR)

        # Diccionario para mapear las descripciones reales para los títulos del ploteo
        task_descriptions = {
            1: {1: "Pick Banana -> Basket -> Cap", 2: "Pick Ycbgelatin -> Basket -> Cap", 3: "Pick Carrot -> Basket -> Cap", 4: "Pick Strawberry -> Basket -> Cap", 5: "Place glass -> Fill juice", 6: "Pick glass -> Basket -> Cap"},
            2: {1: "Syringe -> Bin, Sunscreen -> Bin", 2: "Sunscreen -> Tray, Syrup -> Bin", 3: "Syringe -> Tray, Syrup -> Tray", 4: "Syrup -> Bin, Sunscreen -> Bin", 5: "Sunscreen -> Bin, Syringe -> Tray", 6: "Syrup -> Bin, Syringe -> Bin"},
            3: {1: "Pack right/left bottles behind/front", 2: "Pack dominosugar, YcbMustardBottle", 3: "Pack kelloges, dominosugar", 4: "Pack YcbMustardBottle, kelloges", 5: "Pack dominosugar, right bottle(behind)", 6: "Pack right bottle(front), YcbMustardBottle"}
        }

        for scene in range(1, 4):
            for task in range(1, 7):
                print(f"\n=========================================")
                print(f" Procesando Scene {scene} - Task {task}")
                print(f"=========================================")
                
                # Listas exclusivas para acumular las 10 demostraciones de ESTA TAREA
                task_demos_to_plot = []
                task_segments_to_plot = []
                
                task_output_dir = os.path.join(OUTPUT_BASE_DIR, f"Scene_{scene}_Task_{task}")
                if not os.path.exists(task_output_dir):
                    os.makedirs(task_output_dir)

                for demo_idx in range(1, 11):
                    # Según tu carpeta: Escena.Tarea.Demo_idx (ej: 1.1.1 hasta 1.1.10)
                    folder_name = f"{scene}.{task}.{demo_idx}"
                    csv_path = os.path.join(LHT_BASE_DIR, folder_name, 'data.csv')
                    
                    if not os.path.exists(csv_path):
                        continue
                        
                    print(f" -> Analizando Demo {demo_idx}...")
                    try:
                        segments_list, demo_traj = main_lht(csv_path)
                        
                        # plot=False aquí para que no salten picos intermedios
                        #print(f"    Segmentos detectados (antes de combinar): {segments_list}")
                        final_segments = probabilistically_combine(segments_list, len(demo_traj), window_size=1, n_samples=3, n_pass=2, plot=False)
                        
                        # Guardar los submovimientos resultantes en txt
                        for i in range(len(final_segments) - 1):
                            start_idx = int(final_segments[i])
                            end_idx = int(final_segments[i+1])
                            segment_traj = demo_traj[start_idx:end_idx, :]
                            
                            filename = f'demo_{demo_idx}_segmento_{i+1}.txt'
                            file_path = os.path.join(task_output_dir, filename)
                            with open(file_path, 'w') as f:
                                np.savetxt(f, segment_traj, comments='', fmt='%.6f')
                                
                        # Añadir a las listas de la tarea actual para dibujarlas juntas
                        task_demos_to_plot.append(demo_traj)
                        task_segments_to_plot.append(final_segments)
                                
                    except Exception as e:
                        print(f"   [Error] Falló el procesamiento de {folder_name}: {e}")

                # =======================================================
                # PLOTEO AL FINAL DE LA TAREA (Se ejecuta 1 vez por Tarea)
                # =======================================================
                if len(task_demos_to_plot) > 0:
                    task_desc = task_descriptions.get(scene, {}).get(task, f"Tarea {task}")
                    print(f"\n[Visualización] Renderizando Tarea: {task_desc}")
                    
                    plt.rcParams['figure.figsize'] = (10, 8)
                    fig = plt.figure()
                    fig.suptitle(f'Scene {scene} | Task {task}: {task_desc}')
                    ax = plt.axes(projection='3d')
                    
                    # Draw all the trajectories
                    for d_idx in range(len(task_demos_to_plot)):
                        d_traj = task_demos_to_plot[d_idx]
                        d_segs = task_segments_to_plot[d_idx]
                        
                        for i in range(len(d_segs) - 1):
                            start_idx = int(d_segs[i])
                            end_idx = int(d_segs[i+1])
                            segment_traj = d_traj[start_idx:end_idx, :]
                            
                            # --- NUEVO: FILTRO DE SUAVIZADO PARA VISUALIZACIÓN ---
                            
                            window_length = 21 
                            poly_order = 3     # Orden del polinomio (3 suele ser ideal para trayectorias)
                            
                            if len(segment_traj) > window_length:
                                # Aplicamos el filtro a lo largo del eje temporal (axis=0) para X, Y, Z
                                segment_traj_smooth = savgol_filter(segment_traj, window_length, poly_order, axis=0)
                            else:
                                # Si el segmento es muy corto, lo dibujamos sin filtrar
                                segment_traj_smooth = segment_traj
                            # -----------------------------------------------------

                            # Solo añadimos etiqueta en la primera demo para que la leyenda no se repita
                            label_name = f"Segmento {i+1}" if d_idx == 0 else ""
                            
                            # IMPORTANTE: Cambiamos segment_traj por segment_traj_smooth
                            ax.plot3D(segment_traj_smooth[:, 0], segment_traj_smooth[:, 1], segment_traj_smooth[:, 2], 
                                    c=colors[i % len(colors)], label=label_name, lw=2, alpha=0.5)

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.legend()  # Mostrar la leyenda en el gráfico
                    
                    # Se pausará aquí. Cuando cierres la ventana, pasará a procesar la Task+1
                    plt.show() 

        print("\n¡Procesamiento LHT completado! Todos los segmentos estructurados en la carpeta LHT_SegmentsFolder_pruebas/")

    else:
        print("Modo no implementado. Por favor, selecciona 'LHT' para procesar las trayectorias del simulador Franka LHT.")