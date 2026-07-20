import pybullet as p
import time
try:
    from . import config
except ImportError:
    import config
#from kitchen_assets.load_franka_kitchen import loadFrankaKitchen, updateFrankaKitchen
try:
    from .kitchen_assets.load_franka_kitchen import loadFrankaKitchen, updateFrankaKitchen
except ImportError:
    from kitchen_assets.load_franka_kitchen import loadFrankaKitchen, updateFrankaKitchen

class Environment:

    def __init__(self):
        self.value = None

    def load(self):
        p.resetDebugVisualizerCamera(cameraDistance=config.cameraDistance,
                                     cameraYaw=config.cameraYaw,
                                     cameraPitch=config.cameraPitch,
                                     cameraTargetPosition=config.cameraTargetPosition)
        kitchen, kettle = loadFrankaKitchen()
        
        self.value = kitchen      
        self.kitchen_id = kitchen 
        self.kettle_id = kettle   
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def update(self):
        updateFrankaKitchen(self.value)
        p.stepSimulation()
        time.sleep(config.control_dt)
