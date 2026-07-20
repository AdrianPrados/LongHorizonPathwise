import pybullet as p
import pybullet_data
import time
from env import Environment
from robot import Robot
from utils.parse_mjl import parse_mjl_logs
import math

def replay_trajectory(mjl_path):
    # ==========================================
    # 1. INICIAR PYBULLET CON INTERFAZ GRÁFICA
    # ==========================================
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")

    # ==========================================
    # 2. CARGAR EL ENTORNO Y EL ROBOT
    # ==========================================
    env = Environment()
    env.load()
    robot = Robot()

    # ==========================================
    # 3. LEER LA TRAYECTORIA DEL .MJL
    # ==========================================
    print("Cargando trayectoria desde el archivo...")
    mjl_data = parse_mjl_logs(mjl_path, skipamount=1)
    qpos_trajectory = mjl_data['qpos']
    
    movable_joints = []
    print("\n--- ORDEN DE ARTICULACIONES EN PYBULLET ---")
    for j in range(p.getNumJoints(robot.id)):
        info = p.getJointInfo(robot.id, j)
        joint_type = info[2]
        
        # Si es un motor móvil (rotatorio o prismático)
        if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
            movable_joints.append(j)
            
            # Extraer y decodificar los nombres
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            
            print(f"Índice: {j} | Joint: {joint_name} | Link: {link_name}")
    print("-------------------------------------------\n")

    print("¡Todo listo! Iniciando reproducción en 3 segundos...")
    time.sleep(3) # Pausa para que te dé tiempo a ver cómo empieza

    # ==========================================
    # 4. BUCLE DE EJECUCIÓN (REPLAY)
    # ==========================================
    for qpos in qpos_trajectory:
        # Los primeros 9 valores de la matriz corresponden al brazo Franka
        robot_qpos = qpos[:9]
        #print("Reproduciendo qpos:", robot_qpos)
        offset_muneca = math.pi / 2  # 45 grados en radianes (0.785398...)
        robot_qpos[5] = robot_qpos[5] + offset_muneca

        # Aplicar control de posición a cada motor del robot
        for i, joint_idx in enumerate(movable_joints):
            if i < len(robot_qpos):
                p.setJointMotorControl2(
                    bodyUniqueId=robot.id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL, # Usamos los motores físicos
                    targetPosition=robot_qpos[i],
                    force=500,       # Fuerza alta para asegurar que imita bien el movimiento
                    maxVelocity=9.0  # Velocidad máxima permitida
                )
        
        # Avanzar un paso en la simulación (tu env.py ya incluye el p.stepSimulation() y el delay)
        env.update()

    print("Reproducción de la trayectoria terminada.")
    
    # ==========================================
    # 5. MANTENER LA SIMULACIÓN ABIERTA AL ACABAR
    # ==========================================
    while True:
        env.update()

if __name__ == "__main__":
    archivo_mjl = "kitchen_demos_multitask/postcorl_microwave_bottomknob_switch_slide/kitchen_playdata_2019_07_11_13_32_19.mjl"
    replay_trajectory(archivo_mjl)