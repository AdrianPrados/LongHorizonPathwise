import os
import glob
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from robot import Robot
from utils.parse_mjl import parse_mjl_logs
import math


def extract_trajectories_pybullet(mjl_path, ee_link_name="panda_link8"):
    """
    Lee un archivo .mjl y usa PyBullet para calcular la trayectoria (x,y,z) y 
    la orientación del efector final.
    """
    mjl_data = parse_mjl_logs(mjl_path, skipamount=1)
    qpos_trajectory = mjl_data['qpos']
    
    # Modo DIRECT para no abrir GUI y procesar datos a máxima velocidad sin colapsar el PC
    physicsClient = p.connect(p.DIRECT) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    robot = Robot() 
    
    movable_joints = []
    ee_link_index = -1
    
    for j in range(p.getNumJoints(robot.id)):
        info = p.getJointInfo(robot.id, j)
        joint_type = info[2]
        link_name = info[12].decode('utf-8')
        
        if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
            movable_joints.append(j)
            
        if link_name == ee_link_name:
            ee_link_index = j
            
    if ee_link_index == -1:
        ee_link_index = 8 
        print(f"No se encontró {ee_link_name}, usando el link {ee_link_index} por defecto.")

    ee_positions = []
    ee_orientations = []

    for qpos in qpos_trajectory:
        # Convertimos a lista para poder modificar el valor
        robot_qpos = list(qpos[:9])
        
        # --- CORRECCIÓN: AÑADIR EL OFFSET DE LA MUÑECA ---
        offset_muneca = math.pi / 2
        robot_qpos[5] = robot_qpos[5] + offset_muneca
        # --------------------------------------------------
        
        for i, joint_idx in enumerate(movable_joints):
            if i < len(robot_qpos):
                p.resetJointState(robot.id, joint_idx, robot_qpos[i])
        
        link_state = p.getLinkState(robot.id, ee_link_index)
        
        ee_positions.append(link_state[0])
        ee_orientations.append(link_state[1])

    p.disconnect(physicsClient)
    
    return np.array(ee_positions), np.array(ee_orientations)

# ==========================================
# 2. FUNCIÓN DE PROCESAMIENTO EN LOTE
# ==========================================
def process_all_mjl_files(input_folder, output_folder):
    """
    Busca todos los .mjl, extrae la trayectoria, guarda los .txt
    y devuelve una lista con todas las posiciones 3D para plotearlas.
    """
    search_pattern = os.path.join(input_folder, '**', '*.mjl')
    mjl_files = glob.glob(search_pattern, recursive=True)
    
    todas_las_posiciones = []

    if not mjl_files:
        print(f"No se encontraron archivos .mjl en: {input_folder}")
        return todas_las_posiciones

    print(f"Se han encontrado {len(mjl_files)} archivos .mjl para procesar.\n")

    for mjl_path in mjl_files:
        print(f"Procesando: {mjl_path}")
        try:
            # 1. Extraer los datos
            posiciones, orientaciones = extract_trajectories_pybullet(mjl_path)
            
            # Guardamos las posiciones para el plot global
            todas_las_posiciones.append(posiciones)
            
            # 2. Concatenar y guardar en .txt
            trayectoria_completa = np.hstack((posiciones, orientaciones))
            rel_path = os.path.relpath(mjl_path, input_folder)
            txt_filename = os.path.splitext(rel_path)[0] + ".txt"
            out_path = os.path.join(output_folder, txt_filename)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savetxt(out_path, trayectoria_completa, fmt='%.6f', header='x y z qx qy qz qw', comments='')
            
        except Exception as e:
            print(f"  -> [ERROR] Fallo al procesar {mjl_path}: {e}")

    return todas_las_posiciones

# ==========================================
# 3. FUNCIÓN DE PLOTEO GLOBAL
# ==========================================
def plot_all_trajectories(all_positions):
    """
    Recibe una lista de arrays Nx3 y los plotea todos en el mismo gráfico 3D.
    """
    if not all_positions:
        return
        
    print("\nGenerando gráfico 3D global...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujamos cada trayectoria almacenada
    for i, pos in enumerate(all_positions):
        # Línea de la trayectoria (transparencia para no saturar si hay muchas)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.6, linewidth=1.5)
        
        # Puntos de inicio (verde) y fin (rojo)
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='green', s=15, marker='o')
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='red', s=15, marker='x')
        
    ax.set_xlabel('Posición X (m)')
    ax.set_ylabel('Posición Y (m)')
    ax.set_zlabel('Posición Z (m)')
    ax.set_title(f'Conjunto de {len(all_positions)} Demos (Efector Final)')
    ax.grid(True)
    
    plt.show()

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    
    CARPETA_ENTRADA = "kitchen_demos_multitask/postcorl_microwave_kettle_switch_hinge"
    CARPETA_SALIDA = "demos_trajs/postcorl_microwave_kettle_switch_hinge"
    
    print("Iniciando extracción masiva...")
    lista_posiciones_3d = process_all_mjl_files(CARPETA_ENTRADA, CARPETA_SALIDA)
    
    print("\n¡Extracción finalizada!")
    
    # Mostrar todas las demostraciones juntas
    plot_all_trajectories(lista_posiciones_3d)