import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from matplotlib import cm 

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

import matplotlib as mpl
mpl.rc('font', size=11) 

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'] * 5

# =====================================================================
# FUNCIONES MATEMÁTICAS Y DE SEGMENTACIÓN 
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

# =====================================================================
# FUNCIONES DE CLUSTERING Y PLOTEO ANALÍTICO
# =====================================================================
def cluster_segments_got_mock(all_segments):
    if len(all_segments) == 0:
        return []
    
    features = []
    for s in all_segments:
        if len(s) > 0:
            feat = np.hstack((np.mean(s, axis=0), len(s)))
            features.append(feat)
        else:
            features.append(np.zeros(4))
            
    features = np.array(features)
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
    clustering = DBSCAN(eps=0.1, min_samples=1).fit(features)
    return clustering.labels_

def plot_smooth_segment(ax, segment_traj, color, label_name, is_2d=False, alpha=0.8):
    window_length = 21 
    poly_order = 3
    if len(segment_traj) > window_length:
        segment_traj_smooth = savgol_filter(segment_traj, window_length, poly_order, axis=0)
    else:
        segment_traj_smooth = segment_traj
        
    if is_2d:
        ax.plot(segment_traj_smooth[:, 0], segment_traj_smooth[:, 1], c=color, label=label_name, lw=2, alpha=alpha)
    else:
        ax.plot3D(segment_traj_smooth[:, 0], segment_traj_smooth[:, 1], segment_traj_smooth[:, 2], 
                  c=color, label=label_name, lw=2, alpha=alpha)

def plot_gmm_density_2d(ax, demo_traj, segment_indices, title):
    x_min, x_max = np.min(demo_traj[:, 0]) - 0.05, np.max(demo_traj[:, 0]) + 0.05
    y_min, y_max = np.min(demo_traj[:, 1]) - 0.05, np.max(demo_traj[:, 1]) + 0.05
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    XX = np.array([X.ravel(), Y.ravel()]).T
    
    Z_acumulado = np.zeros(X.shape)
    
    segs = list(segment_indices)
    if len(segs) > 0:
        if segs[0] != 0: segs.insert(0, 0)
        if segs[-1] != len(demo_traj): segs.append(len(demo_traj))
        
    for i in range(len(segs) - 1):
        start, end = int(segs[i]), int(segs[i+1])
        seg_data = demo_traj[start:end, :2]
        
        if len(seg_data) > 4:
            gmm = GaussianMixture(n_components=1, covariance_type='full', reg_covar=0.0005, random_state=42)
            gmm.fit(seg_data)
            Z = -gmm.score_samples(XX)
            Z_prob = np.exp(-Z).reshape(X.shape)
            
            Z_acumulado += Z_prob / (np.max(Z_prob) + 1e-6)
            
            ax.contour(X, Y, Z_prob, levels=2, colors=[colors[i % len(colors)]], linewidths=1.8, alpha=0.8)
            
    c = ax.contourf(X, Y, Z_acumulado, levels=40, cmap='viridis', alpha=0.7)
    
    window_length = 21
    poly_order = 3
    if len(demo_traj) > window_length:
        smooth_path = savgol_filter(demo_traj[:, :2], window_length, poly_order, axis=0)
    else:
        smooth_path = demo_traj[:, :2]
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], 'k-', lw=2, alpha=0.6, label="Trajectory Path")
    
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    return c

def plot_got_sequence(fig, row_idx, segA, segB, labelA, labelB):
    target_pts = 100
    t_A = np.linspace(0, 1, len(segA))
    t_B = np.linspace(0, 1, len(segB))
    t_target = np.linspace(0, 1, target_pts)
    
    segA_res = np.zeros((target_pts, 3))
    segB_res = np.zeros((target_pts, 3))
    for i in range(3):
        segA_res[:, i] = np.interp(t_target, t_A, segA[:, i])
        segB_res[:, i] = np.interp(t_target, t_B, segB[:, i])
        
    got_cost = np.mean(np.linalg.norm(segA_res - segB_res, axis=1))

    all_pts = np.vstack((segA, segB))
    x_min, x_max = np.min(all_pts[:,0]) - 0.05, np.max(all_pts[:,0]) + 0.05
    y_min, y_max = np.min(all_pts[:,1]) - 0.05, np.max(all_pts[:,1]) + 0.05

    ax3d = fig.add_subplot(2, 6, row_idx * 6 + 1, projection='3d')
    ax3d.scatter(segA[:,0], segA[:,1], segA[:,2], c='orange', alpha=0.3, s=10)
    ax3d.scatter(segB[:,0], segB[:,1], segB[:,2], c='lightblue', alpha=0.3, s=10)
    ax3d.scatter(np.mean(segA[:,0]), np.mean(segA[:,1]), np.mean(segA[:,2]), c='red', marker='+', s=100)
    ax3d.scatter(np.mean(segB[:,0]), np.mean(segB[:,1]), np.mean(segB[:,2]), c='blue', marker='+', s=100)
    
    ax3d.set_title(f"(a) 3D: {labelA} vs {labelB}\nEst. GOT Cost: {got_cost:.4f}")

    t_values = [1.0, 0.7, 0.5, 0.3, 0.0] 
    titles = ['GP_1 (t=1.0)', 't=0.7', 't=0.5', 't=0.3', 'GP_0 (t=0.0)']

    for i, t_val in enumerate(t_values):
        ax = fig.add_subplot(2, 6, row_idx * 6 + 2 + i)
        seg_t = (1 - t_val) * segA_res + t_val * segB_res
        heatmap, xedges, yedges = np.histogram2d(
            seg_t[:, 1], seg_t[:, 0], 
            bins=30, range=[[y_min, y_max], [x_min, x_max]]
        )
        heatmap_smooth = gaussian_filter(heatmap, sigma=1.2)
        ax.imshow(heatmap_smooth, origin='lower', cmap='viridis', extent=[x_min, x_max, y_min, y_max])
        ax.set_title(titles[i])
        if i == 0:
            ax.set_ylabel("Interpolation (b)", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])


# =====================================================================
# FUNCIÓN DE LECTURA H5 3D
# =====================================================================
def main_h5_3d(fname):
    """
    Lee datos 3D de los ficheros H5, hace la segmentación heurística
    y devuelve el array de cortes crudos y la lista para la fusión.
    """
    joint_data, tf_data, wrench_data, gripper_data = read_robot_data(fname)
    
    traj_time = tf_data[0][:, 0] + tf_data[0][:, 1] * (10.0**-9)
    traj_pos = tf_data[1]
    
    gripper_time = gripper_data[0][:, 0] + gripper_data[0][:, 1] * (10.0**-9)
    gripper_pos = gripper_data[1]
    
    # Downsampling
    traj_pos, ds_inds = ds.DouglasPeuckerPoints2(traj_pos, 1000)
    traj_time = traj_time[ds_inds]
    gripper_time = gripper_time[ds_inds]
    gripper_pos = gripper_pos[ds_inds]
    
    thresh = 0.25
    ssize = 32
    wsize = 64
    gthresh = 32
    
    traj_segments = segment(traj_time, traj_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, mode="variations")
    gripper_segments = segment(gripper_time, gripper_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, mode="variations")
    
    segments_to_combine = [traj_segments, gripper_segments]
    return traj_segments, segments_to_combine, traj_pos


# =====================================================================
# EJECUCIÓN PRINCIPAL CON PLOTS ANALÍTICOS
# =====================================================================
if __name__ == '__main__':
    
    print("\n=========================================")
    print(" Procesando ficheros .h5 en 3D (Robot Real)")
    print("=========================================")
    
    OUTPUT_BASE_DIR = 'H5_3D_SegmentsFolder'
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
        
    task_demos_to_plot = []
    task_heuristic_to_plot = [] 
    task_fusion_to_plot = []    
    all_raw_segments_for_got = [] 
    
    # Lista de ficheros .h5 a procesar
    h5_files = ['../h5 files/IzquierdaLongLatasDemo.h5'] 
    
    for demo_idx, fname in enumerate(h5_files):
        print(f" -> Analizando Fichero H5 {fname}...")
        try:
            # 1. Obtenemos los cortes crudos (heurística) y la trayectoria
            heuristic_traj, segments_list, demo_traj = main_h5_3d(fname)
            
            # 2. Hacemos la fusión probabilística
            final_segments = probabilistically_combine(segments_list, len(demo_traj), window_size=1, n_samples=3, n_pass=2, plot=False)
            
            # 3. Guardamos y extraemos fragmentos
            for i in range(len(final_segments) - 1):
                start_idx = int(final_segments[i])
                end_idx = int(final_segments[i+1])
                segment_traj = demo_traj[start_idx:end_idx, :]
                
                filename = f'h5demo_{demo_idx+1}_segmento_{i+1}.txt'
                np.savetxt(os.path.join(OUTPUT_BASE_DIR, filename), segment_traj, comments='', fmt='%.6f')
                
                # Para los plots de GOT, acumulamos los segmentos de la primera demo
                if demo_idx == 0: 
                    all_raw_segments_for_got.append(segment_traj)
                    
            task_demos_to_plot.append(demo_traj)
            task_heuristic_to_plot.append(heuristic_traj)
            task_fusion_to_plot.append(final_segments)
            
        except Exception as e:
            print(f"   [Error] Falló el procesamiento de {fname}: {e}")

    # =======================================================
    # PLOTS ANALÍTICOS (IDÉNTICOS AL CÓDIGO ORIGINAL LHT)
    # =======================================================
    if len(task_demos_to_plot) > 0:
        
        # --- FIGURA 1: VISUALIZACIÓN GEOMÉTRICA (1x3) ---
        fig1 = plt.figure(figsize=(18, 5))
        fig1.suptitle('Geometric Pipeline | H5 Real Robot Trajectory', fontsize=14, fontweight='bold')
        
        # 1. Heurística (Cortes crudos)
        ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
        ax1.set_title("1. Kinematic Heuristic (Over-segmented)")
        for d_idx, d_traj in enumerate(task_demos_to_plot):
            if d_idx != 0: continue 
            
            h_segs = task_heuristic_to_plot[d_idx]
            if len(h_segs)>0 and h_segs[0] != 0: h_segs.insert(0, 0)
            if len(h_segs)>0 and h_segs[-1] != len(d_traj): h_segs.append(len(d_traj))
            for i in range(len(h_segs) - 1):
                s_traj = d_traj[int(h_segs[i]):int(h_segs[i+1]), :]
                plot_smooth_segment(ax1, s_traj, colors[i % len(colors)], "")

        # 2. Fusión (Librerías Extraídas)
        ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
        ax2.set_title("2. Probabilistic Segment Fusion")
        for d_idx, d_traj in enumerate(task_demos_to_plot):
            if d_idx != 0: continue
                
            f_segs = task_fusion_to_plot[d_idx]
            for i in range(len(f_segs) - 1):
                s_traj = d_traj[int(f_segs[i]):int(f_segs[i+1]), :]
                lbl = f"Primitive {i+1}"
                plot_smooth_segment(ax2, s_traj, colors[i % len(colors)], lbl)
        ax2.legend()

        # 3. Clustering GOT Mock
        ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
        ax3.set_title("3. Energy-Based Libraries (GOT Clustering)")
        library_labels = cluster_segments_got_mock(all_raw_segments_for_got)
        plotted_libs = set()
        for seg_traj, lib_id in zip(all_raw_segments_for_got, library_labels):
            col = colors[lib_id % len(colors)]
            if lib_id not in plotted_libs:
                lbl = f"Library {lib_id}"
                plotted_libs.add(lib_id)
            else:
                lbl = ""
            plot_smooth_segment(ax3, seg_traj, col, lbl)
        ax3.legend()

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        plt.tight_layout()

        # --- FIGURA 2: ANÁLISIS GMM ---
        fig2 = plt.figure(figsize=(15, 6))
        fig2.suptitle('GMM Spatial Fusion Analysis (H5 Demo)', fontsize=14, fontweight='bold')
        
        demo1_traj = task_demos_to_plot[0]
        
        h_cuts_d1 = list(task_heuristic_to_plot[0])
        ax_gmm1 = fig2.add_subplot(1, 2, 1)
        c1 = plot_gmm_density_2d(ax_gmm1, demo1_traj, h_cuts_d1, 
                                 title=f"Initial State: Fragmented Gaussians")
        fig2.colorbar(c1, ax=ax_gmm1, label="Relative Density")

        f_cuts_d1 = list(task_fusion_to_plot[0])
        ax_gmm2 = fig2.add_subplot(1, 2, 2)
        c2 = plot_gmm_density_2d(ax_gmm2, demo1_traj, f_cuts_d1, 
                                 title=f"Fused State: Extracted Primitives")
        fig2.colorbar(c2, ax=ax_gmm2, label="Relative Density")
        
        plt.tight_layout()

        # --- FIGURA 3: ANÁLISIS GOT ---
        if len(all_raw_segments_for_got) >= 3:
            fig3 = plt.figure(figsize=(18, 7))
            fig3.suptitle('GOT Energy Cost Analysis: Displacement Interpolation', fontsize=16, fontweight='bold')
            
            segA = all_raw_segments_for_got[0]
            segB = all_raw_segments_for_got[1]
            segC = all_raw_segments_for_got[2]

            plot_got_sequence(fig3, row_idx=0, segA=segA, segB=segB, labelA="Seg 1", labelB="Seg 2")
            plot_got_sequence(fig3, row_idx=1, segA=segA, segB=segC, labelA="Seg 1", labelB="Seg 3")
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.3)
        else:
            print("No hay suficientes segmentos en este archivo .h5 para mostrar la matriz 2x6 de interpolación GOT.")
            
    plt.show()