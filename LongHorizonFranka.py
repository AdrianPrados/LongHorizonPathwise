import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates

# Tu librería local
from ProGP import ProGpMp

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'cyan']

def load_segment_data(base_dir, scene, task, demo, segment_idx, gap=10):
    folder_path = os.path.join(base_dir, f"Scene_{scene}_Task_{task}")
    file_name = f"demo_{demo}_segmento_{segment_idx}.txt"
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
    pos_data_full = np.loadtxt(file_path)
    pos_data = pos_data_full[::gap, :]
    dt = 0.01 
    
    num_points = pos_data.shape[0]
    t_data = np.arange(0, num_points * dt, dt).reshape(-1, 1)
    return t_data, pos_data, dt

def chain_gp_pathwise_segments(scene=1, task=1, demo=1, pkl_path='location.pkl', manual_objects=None, dist_threshold=0.15):
    base_dir = '/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Segmentation-and-Grouping-Model/SegmentationProb/scripts/LHT_SegmentsFolder' 
    
    print(f"--- Procesando Cadena S{scene} | T{task} | Demo {demo} ---")

    if not os.path.exists(pkl_path):
        print(f"Advertencia: Archivo de objetos '{pkl_path}' no encontrado.")
        object_loc_dict = {}
    else:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        object_loc_dict = pkl_data.get('object_loc_dict', {})
        print(f"-> Objetos cargados correctamente desde el .pkl: {list(object_loc_dict.keys())}")

    previous_end_pos = None
    cumulative_time = 0.0
    
    all_t, all_prior, all_corrected = [], [], []
    all_raw_pos, all_vias_cond, all_via_times = [], [], []
    used_objects_positions = {}

    segment_idx = 1
    while True:
        try:
            X, Y, dt = load_segment_data(base_dir, scene, task, demo, segment_idx)
            min_len = min(X.shape[0], Y.shape[0])
            X = X[:min_len]
            Y = Y[:min_len]
            size = X.shape[0]
        except FileNotFoundError:
            if segment_idx == 1:
                print("No se encontraron segmentos para esta configuración.")
            else:
                print(f"-> Fin de la cadena. Se procesaron {segment_idx - 1} segmentos.")
            break
            
        print(f"\n>> Aprendiendo Segmento {segment_idx}...")
        
        target_t = X[-1, 0]
        orig_start = Y[0, :]
        orig_end = Y[-1, :]
        
        via_t = np.array([X[0, 0], target_t]).reshape(-1, 1)
        vias_orig = np.array([orig_start, orig_end])
        
        if previous_end_pos is None:
            curr_start = orig_start
        else:
            curr_start = previous_end_pos
            
        curr_end = orig_end
        previous_end_pos = curr_end
        
        gp_mp = ProGpMp(X, Y, via_t, vias_orig, dim=3, demos=1, size=size, observation_noise=0.01)
        gp_mp.BlendedGpMp(gp_mp.ProGP)
        
        test_x = np.copy(X[:, 0]).reshape(-1, 1)
        mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x)

        var_blended[0] = np.where(var_blended[0] < 0, 0, var_blended[0])
        var_blended[1] = np.where(var_blended[1] < 0, 0, var_blended[1])
        var_blended[2] = np.where(var_blended[2] < 0, 0, var_blended[2])

        # 5. Pathwise Optimization
        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        base_kernels = [cls(lengthscales=0.5) for cls in BaseClasses] 
        kernel = kernels.SeparateIndependent(base_kernels)

        mean_gmp_arr = np.vstack(mean_gmp) 
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

        via_t_list = [via_t[0, 0]]
        vias_cond_list = [curr_start]

        if manual_objects:
            for obj_name in manual_objects:
                if obj_name in object_loc_dict:
                    
                    obj_data = object_loc_dict[obj_name]
                    if isinstance(obj_data, (list, tuple)) and len(obj_data) > 0 and isinstance(obj_data[0], (list, tuple, np.ndarray)):
                        obj_pos = np.array(obj_data[0]).flatten()[:3]
                    else:
                        obj_pos = np.array(obj_data).flatten()[:3]
                    
                    distances = np.linalg.norm(Y - obj_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    min_dist = distances[closest_idx]
                    
                    print(f"   [DEBUG] Distancia a '{obj_name}' en Segmento {segment_idx}: {min_dist:.3f} metros")
                    if min_dist <= dist_threshold:
                        closest_time = X[closest_idx, 0]
                        while closest_time in via_t_list or closest_time == target_t:
                            closest_time += 1e-4  
                        
                        via_t_list.append(closest_time)
                        vias_cond_list.append(obj_pos)
                        used_objects_positions[obj_name] = obj_pos
                        print(f"   => [¡ACEPTADO!] Añadido via-point para '{obj_name}' en t={closest_time:.4f}s")

        via_t_list.append(via_t[-1, 0])
        vias_cond_list.append(curr_end)

        sort_indices = np.argsort(via_t_list)
        via_t_cond = np.array(via_t_list)[sort_indices]
        vias_cond = np.array(vias_cond_list)[sort_indices]

        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_cond, dtype=default_float()), axis=0)

        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        F_obs = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 

        t0 = time.time()
        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
        X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 
        dF_full = dF_fn(X_test)
        F_cond = F_prior_full + dF_full
        t1 = time.time()
        print(f"Tiempo pathwise update Seg {segment_idx}: {t1 - t0:.6f} segundos")

        mean_corrected = np.squeeze(F_cond.numpy())
        mean_prior = np.squeeze(mean_gmp_arr.T)
        
        t_shifted = test_x + cumulative_time
        via_t_shifted = via_t_cond + cumulative_time
        
        all_t.append(t_shifted)
        all_prior.append(mean_prior)
        all_corrected.append(mean_corrected)
        all_raw_pos.append(Y)
        all_vias_cond.append(vias_cond)
        all_via_times.append(via_t_shifted)
        
        cumulative_time += target_t 
        segment_idx += 1

    # Visualización...
    num_segments_processed = segment_idx - 1
    if num_segments_processed > 0:
        fig = plt.figure(figsize=(12, 8))
        ax3 = fig.add_subplot(111, projection='3d')

        for i in range(num_segments_processed):
            ax3.plot(all_prior[i][:, 0], all_prior[i][:, 1], all_prior[i][:, 2], '--', linewidth=2, c='grey', alpha=0.4)
            ax3.plot(all_corrected[i][:, 0], all_corrected[i][:, 1], all_corrected[i][:, 2], '-', linewidth=3, c=colors[i % len(colors)], label=f'Seg {i+1}')
            ax3.scatter(all_vias_cond[i][:, 0], all_vias_cond[i][:, 1], all_vias_cond[i][:, 2], c='red', marker='D', s=100)

        for obj_name, obj_pos in used_objects_positions.items():
            ax3.scatter(obj_pos[0], obj_pos[1], obj_pos[2], c='gold', marker='*', s=250, edgecolors='black', linewidths=1.5, label=f'Objeto: {obj_name}')

        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('y [m]')
        ax3.set_zlabel('z [m]')
        ax3.legend(fontsize=10)
        plt.show()

if __name__ == '__main__':
    mis_objetos_manuales = ['banana', 'bucket_cap']
    
    chain_gp_pathwise_segments(
        scene=1, 
        task=1, 
        demo=1, 
        pkl_path='/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/Franka-LHT-Simulator/Trajectory/1.1.1/location.pkl', 
        manual_objects=mis_objetos_manuales,
        dist_threshold=0.25
    )