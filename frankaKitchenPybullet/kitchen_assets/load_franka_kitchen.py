import pybullet as p
try:
    from .. import config
except ImportError:
    import config

kitchenStartOrientationQ = p.getQuaternionFromEuler(config.kitchenStartOrientationE)

def loadFrankaKitchen():

    kitchen = p.loadURDF("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/LongTaskRepo/frankaKitchenPybullet/kitchen_assets/kitchen_env_model.urdf", config.kitchenStartPosition, kitchenStartOrientationQ, useFixedBase=True, globalScaling=config.kitchenGlobalScaling)
    kettle = p.loadURDF("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/LongTaskRepo/frankaKitchenPybullet/kitchen_assets/item_assets/kettle.urdf", config.kettleStartPosition, kitchenStartOrientationQ, useFixedBase=False, globalScaling=config.kitchenGlobalScaling)

    # Change textures
    textureMarble = p.loadTexture("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/LongTaskRepo/frankaKitchenPybullet/kitchen_assets/textures/marble1.png")
    textureMetal = p.loadTexture("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/LongTaskRepo/frankaKitchenPybullet/kitchen_assets/textures/metal1.png")

    # Marble counter tops (doesn't seem to work for sink counter top)
    p.changeVisualShape(kitchen, 2, textureUniqueId=textureMarble)

    # Counter top metal
    p.changeVisualShape(kitchen, 3, textureUniqueId=textureMetal)

    # Oven metal
    p.changeVisualShape(kitchen, 5, textureUniqueId=textureMetal)

    # Knobs
    p.changeVisualShape(kitchen, 7, textureUniqueId=textureMetal)
    p.changeVisualShape(kitchen, 10, textureUniqueId=textureMetal)
    p.changeVisualShape(kitchen, 13, textureUniqueId=textureMetal)
    p.changeVisualShape(kitchen, 16, textureUniqueId=textureMetal)

    # Light switch
    p.changeVisualShape(kitchen, 28, textureUniqueId=textureMetal)
    p.changeVisualShape(kitchen, 30, textureUniqueId=textureMetal)

    # Slide door handle
    p.changeVisualShape(kitchen, 41, textureUniqueId=textureMetal)

    # Hinge door handle
    p.changeVisualShape(kitchen, 46, textureUniqueId=textureMetal)
    p.changeVisualShape(kitchen, 49, textureUniqueId=textureMetal)

    # Microwave
    p.changeVisualShape(kitchen, 54, textureUniqueId=textureMetal)
    p.changeVisualShape(kitchen, 57, textureUniqueId=textureMetal)

    # Kettle 
    p.changeVisualShape(kettle, 0, textureUniqueId=textureMetal)

    return kitchen, kettle

def updateFrankaKitchen(kitchen):

    knobs = [6, 9, 12, 15]
    burners = [18, 20, 22, 24]

    lightSwitch = 29
    lightBlock = 31
    lightLink = 32

    # --- UMBRAL DE SENSIBILIDAD ---
    # 0 es apagado, -1.57 es girado al máximo.
    umbral_fuego = -0.8 
    umbral_luz = -0.2

    # ==========================================
    # 1. FOGONES: Efecto Click y Anulación de muelle
    # ==========================================
    for knob, burner in zip(knobs, burners):
        if p.getJointState(kitchen, knob)[0] < umbral_fuego:
            # Encender fuego
            p.setJointMotorControl2(kitchen, burner, p.POSITION_CONTROL, targetPosition=-.009 / 2)
            # EFECTO CLICK: Forzar el mando a la posición final y bloquearlo
            p.setJointMotorControl2(kitchen, knob, p.POSITION_CONTROL, targetPosition=-1.57, force=5)
        else:
            # Apagar fuego
            p.setJointMotorControl2(kitchen, burner, p.POSITION_CONTROL, targetPosition=0)
            # FRICCIÓN LIBRE: Anular el efecto muelle para que no baile solo
            p.setJointMotorControl2(kitchen, knob, p.VELOCITY_CONTROL, targetVelocity=0, force=1)

    # ==========================================
    # 2. INTERRUPTOR LUZ: Efecto Click
    # ==========================================
    if p.getJointState(kitchen, lightSwitch)[0] < umbral_luz:
        p.changeVisualShape(kitchen, lightLink, rgbaColor=[2, 2, 2, 1])
        p.setJointMotorControl2(kitchen, lightBlock, p.POSITION_CONTROL, targetPosition=-.05 / 2)
        p.setJointMotorControl2(kitchen, lightSwitch, p.POSITION_CONTROL, targetPosition=-0.7, force=5)
    else:
        p.changeVisualShape(kitchen, lightLink, rgbaColor=[.1, .1, .1, 1])
        p.setJointMotorControl2(kitchen, lightBlock, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(kitchen, lightSwitch, p.VELOCITY_CONTROL, targetVelocity=0, force=1)

    # ==========================================
    # 3. PUERTAS: Evitar que reboten al soltarlas
    # ==========================================
    # Slide (41) y Hinge (46)
    p.setJointMotorControl2(kitchen, 41, p.VELOCITY_CONTROL, targetVelocity=0, force=2)
    p.setJointMotorControl2(kitchen, 46, p.VELOCITY_CONTROL, targetVelocity=0, force=2)