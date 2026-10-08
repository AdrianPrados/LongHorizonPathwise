import pybullet as p
import pybullet_data
import time
from env import Environment
from robot import Robot

def get_element_position(body_id, link_id):
    """Obtiene la posición (x, y, z) de un objeto base o un link específico."""
    if link_id == -1:
        # Es la base del objeto
        pos, _ = p.getBasePositionAndOrientation(body_id)
    else:
        # Es una parte específica (link) dentro de un objeto articulado
        link_state = p.getLinkState(body_id, link_id)
        pos = link_state[4] # Posición global del marco del link
    return pos

# ==========================================
# 1. INICIALIZACIÓN DE PYBULLET
# ==========================================
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")

env = Environment()
env.load()

robot = Robot()

# ==========================================
# 2. DEFINICIÓN DEL MAPA DE OBJETOS
# ==========================================
KITCHEN_OBJECTS_MAP = {
    'microwave': {'action': 'grasp', 'id': env.kitchen_id, 'link': 54}, 
    'kettle':    {'action': 'grasp', 'id': env.kettle_id,  'link': -1},     
    'light':     {'action': 'grasp', 'id': env.kitchen_id, 'link': 29},     
    'bottomknob':{'action': 'grasp', 'id': env.kitchen_id, 'link': 12}, 
    'topknob':   {'action': 'grasp', 'id': env.kitchen_id, 'link': 6},    
    'hinge':     {'action': 'grasp', 'id': env.kitchen_id, 'link': 46},      
    'slide':     {'action': 'grasp', 'id': env.kitchen_id, 'link': 41},      
    'burner':    {'action': 'grasp', 'id': env.kitchen_id, 'link': 18}      
}

# ==========================================
# 3. BUCLE PRINCIPAL
# ==========================================
while True:
    env.update()
    
    # Imprimimos una cabecera para separar cada lectura
    print("\n" + "="*50)
    print("ESTADO DEL ENTORNO - POSICIONES XYZ")
    print("="*50)
    
    for obj_name, info in KITCHEN_OBJECTS_MAP.items():
        obj_id = info['id']
        link_id = info['link']
        
        # Obtenemos posición
        pos = get_element_position(obj_id, link_id)
        
        # Formateamos y mostramos la información en la terminal
        print(f"Objeto: {obj_name.upper()}")
        print(f"  -> ID Cuerpo: {obj_id} | ID Link: {link_id}")
        print(f"  -> Posición:  X: {pos[0]:.4f} | Y: {pos[1]:.4f} | Z: {pos[2]:.4f}")
        print("-" * 50)
        
    # Esperamos medio segundo para no saturar la terminal y poder leer los valores
    time.sleep(0.5)