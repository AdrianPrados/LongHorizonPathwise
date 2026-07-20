import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN
import time as t
from scipy.signal import find_peaks
import os
import glob

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k']

# =====================================================================
# FUNCIONES MATEMÁTICAS ORIGINALES
# =====================================================================
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
    norm_avg_jerk = avg_jerk / np.max(np.abs(avg_jerk) + 1e-6)
    
    acceleration = calc_acceleration(time, data)
    total_acceleration = signed_norm(acceleration)
    avg_acceleration = moving_average(total_acceleration, window_size)
    norm_avg_acceleration = avg_acceleration / np.max(np.abs(avg_acceleration) + 1e-6)
    
    if mode == "variations":
        segments = detect_brutal_changes(norm_avg_jerk, base_thresh, segment_size, norm_avg_acceleration)
    elif mode == "threshold":
        segments = count_thresh(norm_avg_jerk, base_thresh, segment_size, grace_thresh, norm_avg_acceleration)
    else:
        raise ValueError("Invalid Mode")
    
    for i in range(1, len(segments)):
        segments[i] = segments[i] + window_size // 2
        
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
        if len(segment_list) > 1: # Prevenir fallos con listas vacías
            segment_probabilities = calc_segment_prob(segment_list[1:], data_len, window_size, plot=False)
            probabilities = probabilities * segment_probabilities
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def probabilistically_combine(list_of_list_of_segments, data_len, window_size, n_samples=10, n_pass=2, plot=False):
    probabilities = calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot)
    peaks, _= find_peaks(probabilities)
    
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(probabilities, label="Probabilidades", marker="o", linestyle="-", markersize=2)
        plt.scatter(peaks, probabilities[peaks], color="red", label="Peaks", zorder=5)
        plt.title("Peaks in probabilities distributions (Consenso Global)")
        plt.xlabel("Index")
        plt.ylabel("Probabilities")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()
    
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

# =====================================================================
# FUNCIONES ESPECÍFICAS PARA FRANKA KITCHEN (Archivos .txt)
# =====================================================================
def read_and_normalize_txt(txt_path, target_length=1000):
    """
    Lee los archivos .txt y los normaliza a TARGET_LENGTH (ej: 1000 puntos)
    para que todas las demos de la misma tarea puedan combinarse matemáticamente.
    """
    # skiprows=1 soluciona el error 'x' to float64
    data = np.loadtxt(txt_path, skiprows=1) 
    
    num_frames = len(data)
    dt = 0.01
    time = np.arange(0, num_frames * dt, dt)
    
    new_time = np.linspace(time[0], time[-1], target_length)
    new_data = np.zeros((target_length, data.shape[1]))
    
    # Interpolar cada columna (x, y, z, qx, qy, qz, qw)
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(new_time, time, data[:, i])
        
    traj_pos = new_data[:, 0:3]
    traj_quat = new_data[:, 3:7]
    
    return new_time, traj_pos, traj_quat, new_data

def get_demo_segments(time, traj_pos, traj_quat):
    thresh = 0.25 
    ssize = 32
    wsize = 64
    gthresh = 32
    
    pos_segments = segment(time, traj_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False, mode="variations")
    quat_segments = segment(time, traj_quat, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False, mode="variations")
    
    return [pos_segments, quat_segments]


# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    
    INPUT_BASE_DIR = '/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/frankaKitchenPybullet/demos_trajs'
    OUTPUT_BASE_DIR = 'KitchenFranka_SegmentsFolder'
    TARGET_LENGTH = 1000 # Longitud común para combinar todas las demos

    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    task_folders = [f.path for f in os.scandir(INPUT_BASE_DIR) if f.is_dir()]
    
    if len(task_folders) == 0:
        print(f"No se encontraron carpetas de tareas en {INPUT_BASE_DIR}")
        exit()

    for task_folder in task_folders:
        task_name = os.path.basename(task_folder)
        print(f"\n=========================================")
        print(f" Procesando Tarea Global: {task_name}")
        print(f"=========================================")
        
        task_output_dir = os.path.join(OUTPUT_BASE_DIR, task_name)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)

        txt_files = glob.glob(os.path.join(task_folder, '*.txt'))
        
        all_segments = []
        all_demos = []

        # 1. LEER TODAS LAS DEMOS DE LA TAREA Y ACUMULAR SUS SEGMENTOS
        for demo_idx, txt_path in enumerate(txt_files, start=1):
            demo_name = os.path.basename(txt_path)
            #print(f" -> Analizando Demo {demo_idx} ({demo_name})...")
            try:
                # Normalizamos a 1000 puntos para poder cruzarlas
                time, traj_pos, traj_quat, norm_data = read_and_normalize_txt(txt_path, TARGET_LENGTH)
                
                # Extraemos las propuestas de corte (posición y orientación)
                demo_segments_list = get_demo_segments(time, traj_pos, traj_quat)
                
                # ACUMULAMOS GLOBALMENTE (Como en tu código original .extend)
                all_segments.extend(demo_segments_list)
                all_demos.append(norm_data)
                        
            except Exception as e:
                print(f"   [Error] Falló el procesamiento de {demo_name}: {e}")

        if len(all_demos) == 0:
            continue

        # 2. COMBINAR GLOBALMENTE (Cruza la información de TODAS las demos)
        # Puedes subir n_pass a 3 o 4 si tienes muchas demos y quieres ser más estricto
        n_pases = max(2, len(all_demos) // 3)
        print(f"\n>> Calculando consenso global de segmentos (n_pass={n_pases})...")
        unified_segments = probabilistically_combine(all_segments, TARGET_LENGTH, window_size=1, n_samples=50, n_pass=n_pases, plot=True)
        print(f"   Cortes unificados definitivos: {unified_segments}")

        # 3. GUARDAR LOS DATOS (Cada demo en su propio archivo, pero con los mismos cortes unificados)
        for i in range(len(unified_segments) - 1):
            start_idx = unified_segments[i]
            end_idx = unified_segments[i+1]
            
            for j in range(len(all_demos)):
                # Extraemos este segmento específico de la demo 'j'
                chunk = all_demos[j][start_idx:end_idx, :]
                
                # Lo guardamos de forma independiente, preservando a qué demo pertenece
                filename = f'demo_{j+1}_segmento_{i+1}.txt'
                file_path = os.path.join(task_output_dir, filename)
                
                with open(file_path, 'w') as f:
                    np.savetxt(f, chunk, comments='', fmt='%.6f')
        
        # 4. PLOTEAR TODAS LAS DEMOS SEGMENTADAS CON EL MISMO COLOR POR SEGMENTO
        plt.rcParams['figure.figsize'] = (9, 7)
        fig = plt.figure()
        fig.suptitle(f'Task: {task_name}')
        ax = plt.axes(projection='3d')
        
        for j in range(len(all_demos)):
            demo = all_demos[j]
            for i in range(len(unified_segments)-1):
                start_idx = unified_segments[i]
                end_idx = unified_segments[i+1]
                
                # Dibujamos cada tramo
                ax.plot3D(demo[start_idx:end_idx, 0], 
                          demo[start_idx:end_idx, 1], 
                          demo[start_idx:end_idx, 2], 
                          c=colors[i % len(colors)], 
                          label=f"Segmento {i+1}" if j == 0 else "", 
                          lw=3, alpha=0.6)
                          
        # Limpiar estilo de fondo (como en tu código original)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.legend()
        plt.tight_layout()
        plt.show()

    print("\n¡Procesamiento completado! Todos los segmentos agrupados en KitchenFranka_SegmentsFolder/")