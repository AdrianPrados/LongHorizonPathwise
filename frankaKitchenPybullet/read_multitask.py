import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt  # <--- NUEVA IMPORTACIÓN
from robot import Robot
from utils.parse_mjl import parse_mjl_logs

# ==========================================
# 2. FUNCIÓN DE EXTRACCIÓN EN PYBULLET
# ==========================================
def extract_trajectories_pybullet(mjl_path, ee_link_name="panda_link8"):
    """
    Lee un archivo .mjl y usa PyBullet para calcular la trayectoria (x,y,z) y 
    la orientación del efector final.
    """
    mjl_data = parse_mjl_logs(mjl_path, skipamount=1)
    qpos_trajectory = mjl_data['qpos']
    
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
        robot_qpos = qpos[:9] 
        
        for i, joint_idx in enumerate(movable_joints):
            if i < len(robot_qpos):
                p.resetJointState(robot.id, joint_idx, robot_qpos[i])
        
        link_state = p.getLinkState(robot.id, ee_link_index)
        
        ee_positions.append(link_state[0])
        ee_orientations.append(link_state[1])

    p.disconnect()
    
    return np.array(ee_positions), np.array(ee_orientations)

# ==========================================
# 3. EJECUTAR Y PLOTEAR
# ==========================================
if __name__ == "__main__":
    archivo_mjl = "kitchen_demos_multitask/friday_kettle_bottomknob_hinge_slide/kitchen_playdata_2019_06_28_13_35_48.mjl"
    
    posiciones, orientaciones = extract_trajectories_pybullet(archivo_mjl)
    
    print(f"Trayectorias extraídas correctamente.")
    print(f"Número de pasos (frames): {len(posiciones)}")
    print(f"Posición inicial [x, y, z]: {posiciones[0]}")
    
    # ==========================================
    # 4. PLOTEAR EL PATH EN 3D
    # ==========================================
    print("Generando gráfico 3D...")
    
    # Crear la figura y el eje 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extraer los arrays de X, Y, y Z
    x = posiciones[:, 0]
    y = posiciones[:, 1]
    z = posiciones[:, 2]
    
    # Dibujar la línea de la trayectoria
    ax.plot(x, y, z, label='Trayectoria del Efector Final', color='b', linewidth=2, alpha=0.7)
    
    # Marcar el punto de INICIO (en verde) y el FIN (en rojo) para que se entienda el movimiento
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Inicio', marker='o', zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Fin', marker='X', zorder=5)
    
    # Etiquetas y título
    ax.set_xlabel('Posición X')
    ax.set_ylabel('Posición Y')
    ax.set_zlabel('Posición Z')
    ax.set_title('Trayectoria 3D de la tarea')
    ax.legend()
    
    # Mostrar el gráfico en una ventana interactiva (puedes rotarlo con el ratón)
    plt.show()