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

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates

# Tu librería local
from ProGP import ProGpMp

import matplotlib as mpl
mpl.rc('font', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'cyan']

class InteractiveTrajectoryPlotter:
    """
    Clase para renderizar la trayectoria 3D de todos los segmentos y permitir 
    arrastrar los via-points libremente en el espacio para recalcular 
    el GP Pathwise Conditioning en tiempo real.
    """
    def __init__(self, segments_info, all_prior, all_corrected):
        self.segments_info = segments_info
        self.all_prior = all_prior
        self.lines = []

        self.points = []
        self.point_mapping = [] 
        
        for seg_idx, info in enumerate(segments_info):
            for v_idx, pt in enumerate(info['vias_cond']):
                self.points.append(pt)
                self.point_mapping.append((seg_idx, v_idx))
                
        self.points = np.array(self.points, float)
        self.last_points = self.points.copy()

        # Variables de estado para acumular desplazamientos
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.accum_dz = 0.0

        # Crear figura
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)

        # Plotear el Prior y la Trayectoria Corregida Inicial
        for i in range(len(segments_info)):
            self.ax.plot(all_prior[i][:, 0], all_prior[i][:, 1], all_prior[i][:, 2], 
                         '--', linewidth=2, c='grey', alpha=0.4)
            line, = self.ax.plot(all_corrected[i][:, 0], all_corrected[i][:, 1], all_corrected[i][:, 2], 
                                 '-', linewidth=3, c=colors[i % len(colors)], label=f'Seg {i+1}')
            self.lines.append(line)

        # Determinar colores de los puntos y extraer nombres de objetos únicos
        point_colors = []
        unique_objects = set()
        
        for i in range(len(self.points)):
            seg_idx, v_idx = self.point_mapping[i]
            label = self.segments_info[seg_idx]['obj_names'].get(v_idx, "Nodo")
            if label == "Nodo":
                point_colors.append('blue')
            else:
                point_colors.append('orange')
                unique_objects.add(label)

        # Plotear los scatter points interactivos
        xs, ys, zs = self.points.T
        self.scatter = self.ax.scatter(xs, ys, zs, s=150, c=point_colors, picker=5, marker='D', edgecolors='white', linewidths=1.5)

        # Construir leyenda personalizada
        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Unión/Nodo Vía'))
        for obj_name in unique_objects:
            handles.append(mlines.Line2D([], [], color='orange', marker='D', linestyle='None', markersize=10, label=f'Objeto: {obj_name}'))
        
        self.ax.legend(handles=handles, fontsize=12, loc='best')

        # Anotación flotante para cuando hacemos clic
        self.annot = self.ax.annotate('', xy=(0,0), xytext=(10,10), textcoords='offset points', 
                                      bbox=dict(boxstyle='round,pad=0.3', fc='w', alpha=0.9), 
                                      arrowprops=dict(arrowstyle='->'))
        self.annot.set_visible(False)

        # Eventos
        c = self.fig.canvas
        c.mpl_connect('button_press_event', self.on_press)
        c.mpl_connect('motion_notify_event', self.on_motion)
        c.mpl_connect('button_release_event', self.on_release)
        c.mpl_connect('scroll_event', self.on_scroll)

        self.press = None
        self.ind = None
        self.ax.set_title('Arrastra nodos (Plano XY) | Mantén clic + Scroll (Eje Z)', fontsize=16)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_zlabel('z [m]')

    def on_press(self, event):
        if event.button != 1 or event.inaxes != self.ax: return
        hit, info = self.scatter.contains(event)
        if not hit: return
        
        self.ind = info['ind'][0]
        x0, y0, z0 = self.points[self.ind]
        self.press = (x0, y0, z0, event.x, event.y)
        
        # Reiniciar acumuladores en cada nuevo clic
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.accum_dz = 0.0

        self.update_annotation(event)
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if not self.press or self.ind is None or event.inaxes != self.ax or event.button != 1: return
        x0, y0, z0, xp, yp = self.press
        dx_pix, dy_pix = event.x - xp, event.y - yp
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        w, h = self.ax.bbox.width, self.ax.bbox.height
        
        self.current_dx = dx_pix * (xlim[1]-xlim[0]) / w
        self.current_dy = dy_pix * (ylim[1]-ylim[0]) / h
        
        self.apply_delta(event)

    def on_scroll(self, event):
        if self.ind is None or self.press is None: return
        
        zlim = self.ax.get_zlim()
        # Acumular el paso del scroll de forma continua
        dz_step = event.step * 0.02 * (zlim[1] - zlim[0])
        self.accum_dz += dz_step
        
        self.apply_delta(event)

    def apply_delta(self, event):
        delta = np.array([self.current_dx, self.current_dy, self.accum_dz])
        old_pos = self.last_points[self.ind].copy()
        
        for i, p in enumerate(self.last_points):
            if np.linalg.norm(p - old_pos) < 1e-4:
                self.points[i] = self.last_points[i] + delta
                
                seg_idx, v_idx = self.point_mapping[i]
                self.segments_info[seg_idx]['vias_cond'][v_idx] = self.points[i]
                self.recompute_segment(seg_idx)
        
        xs, ys, zs = self.points.T
        self.scatter._offsets3d = (xs, ys, zs)
        
        self.update_annotation(event)
        self.fig.canvas.draw_idle()

    def update_annotation(self, event):
        seg_idx, v_idx = self.point_mapping[self.ind]
        label = self.segments_info[seg_idx]['obj_names'].get(v_idx, "Nodo")
        new_pos = self.points[self.ind]
        
        self.annot.xy = (event.x, event.y)
        self.annot.set_text(f'{label}\nPos: ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})')

    def on_release(self, event):
        if self.ind is not None:
            self.last_points = self.points.copy()
        self.press = None
        self.ind = None
        self.annot.set_visible(False)
        self.fig.canvas.draw_idle()

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
        
    def show(self):
        plt.show()

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
    
    all_prior, all_corrected = [], []
    segments_info = []

    segment_idx = 1
    while True:
        try:
            X, Y, dt = load_segment_data(base_dir, scene, task, demo, segment_idx)
            min_len = min(X.shape[0], Y.shape[0])
            X = X[:min_len]
            Y = Y[:min_len]
            size = X.shape[0]
        except FileNotFoundError:
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

        BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
        base_kernels = [cls(lengthscales=0.5) for cls in BaseClasses] 
        kernel = kernels.SeparateIndependent(base_kernels)

        mean_gmp_arr = np.vstack(mean_gmp) 
        F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0)

        points_info_list = [(via_t[0, 0], curr_start, "Start")]

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
                    
                    if min_dist <= dist_threshold:
                        closest_time = X[closest_idx, 0]
                        while closest_time in [p[0] for p in points_info_list] or closest_time == target_t:
                            closest_time += 1e-4  
                        
                        points_info_list.append((closest_time, obj_pos, obj_name))
                        print(f"   => [ACEPTADO] Via-point interactivo creado para '{obj_name}'")

        points_info_list.append((via_t[-1, 0], curr_end, "End"))
        points_info_list.sort(key=lambda x: x[0])

        via_t_cond = np.array([p[0] for p in points_info_list])
        vias_cond = np.array([p[1] for p in points_info_list])
        
        obj_names_map = {idx: p[2] for idx, p in enumerate(points_info_list) if p[2] not in ["Start", "End"]}

        X_obs = tf.convert_to_tensor(via_t_cond.reshape(-1, 1), dtype=default_float())
        Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_cond, dtype=default_float()), axis=0)

        obs_idx = [int(np.argmin(np.abs(test_x.flatten() - t))) for t in via_t_cond]
        F_obs = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0) 

        t0 = time.time()
        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
        X_test = tf.convert_to_tensor(test_x, dtype=default_float()) 
        dF_full = dF_fn(X_test)
        F_cond = F_prior_full + dF_full
        print(f"Tiempo pathwise update Seg {segment_idx}: {time.time() - t0:.6f} segundos")

        mean_corrected = np.squeeze(F_cond.numpy())
        mean_prior = np.squeeze(mean_gmp_arr.T)
        
        all_prior.append(mean_prior)
        all_corrected.append(mean_corrected)

        segments_info.append({
            'kernel': kernel,
            'F_prior_full': F_prior_full,
            'via_t_cond': via_t_cond,
            'vias_cond': vias_cond,
            'mean_gmp_arr': mean_gmp_arr,
            'test_x': test_x,
            'obj_names': obj_names_map
        })
        
        segment_idx += 1

    num_segments_processed = segment_idx - 1
    if num_segments_processed > 0:
        print("\n>> Abriendo Visor Interactivo...")
        plotter = InteractiveTrajectoryPlotter(segments_info, all_prior, all_corrected)
        plotter.show()

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