import pybullet as p
try:
    from .. import config
except ImportError:
    import config

kitchenStartOrientationQ = p.getQuaternionFromEuler(config.kitchenStartOrientationE)

def loadFrankaKitchen():

    kitchen = p.loadURDF("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/frankaKitchenPybullet/kitchen_assets/kitchen_env_model.urdf", config.kitchenStartPosition, kitchenStartOrientationQ, useFixedBase=True, globalScaling=config.kitchenGlobalScaling)
    kettle = p.loadURDF("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/frankaKitchenPybullet/kitchen_assets/item_assets/kettle.urdf", config.kettleStartPosition, kitchenStartOrientationQ, useFixedBase=False, globalScaling=config.kitchenGlobalScaling)

    # Change textures
    textureMarble = p.loadTexture("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/frankaKitchenPybullet/kitchen_assets/textures/marble1.png")
    textureMetal = p.loadTexture("/home/adrian/Escritorio/ImitationLearning/LongHorizonSegmentation/frankaKitchenPybullet/kitchen_assets/textures/metal1.png")

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

    # Kettle (wood1.png file doesn't seem to load)
    p.changeVisualShape(kettle, 0, textureUniqueId=textureMetal)

    return kitchen, kettle

def updateFrankaKitchen(kitchen):

    knob1 = 6
    knob2 = 9
    knob3 = 12
    knob4 = 15

    burner1 = 18
    burner2 = 20
    burner3 = 22
    burner4 = 24

    lightSwitch = 29
    lightBlock = 31
    lightLink = 32

    # Manually operate the burners and light
    if p.getJointState(kitchen, knob1)[0] >= -1.57 and p.getJointState(kitchen, knob1)[0] < -1.57 / 2:
        p.setJointMotorControl2(kitchen, burner1, p.POSITION_CONTROL, targetPosition=-.009 / 2)
    elif p.getJointState(kitchen, knob1)[0] >= -1.57 / 2 and p.getJointState(kitchen, knob1)[0] < 0:
        p.setJointMotorControl2(kitchen, burner1, p.POSITION_CONTROL, targetPosition=0)
    if p.getJointState(kitchen, knob2)[0] >= -1.57 and p.getJointState(kitchen, knob2)[0] < -1.57 / 2:
        p.setJointMotorControl2(kitchen, burner2, p.POSITION_CONTROL, targetPosition=-.009 / 2)
    elif p.getJointState(kitchen, knob2)[0] >= -1.57 / 2 and p.getJointState(kitchen, knob2)[0] < 0:
        p.setJointMotorControl2(kitchen, burner2, p.POSITION_CONTROL, targetPosition=0)
    if p.getJointState(kitchen, knob3)[0] >= -1.57 and p.getJointState(kitchen, knob3)[0] < -1.57 / 2:
        p.setJointMotorControl2(kitchen, burner3, p.POSITION_CONTROL, targetPosition=-.009 / 2)
    elif p.getJointState(kitchen, knob3)[0] >= -1.57 / 2 and p.getJointState(kitchen, knob3)[0] < 0:
        p.setJointMotorControl2(kitchen, burner3, p.POSITION_CONTROL, targetPosition=0)
    if p.getJointState(kitchen, knob4)[0] >= -1.57 and p.getJointState(kitchen, knob4)[0] < -1.57 / 2:
        p.setJointMotorControl2(kitchen, burner4, p.POSITION_CONTROL, targetPosition=-.009 / 2)
    elif p.getJointState(kitchen, knob4)[0] >= -1.57 / 2 and p.getJointState(kitchen, knob4)[0] < 0:
        p.setJointMotorControl2(kitchen, burner4, p.POSITION_CONTROL, targetPosition=0)

    if p.getJointState(kitchen, lightSwitch)[0] >= -.7 and p.getJointState(kitchen, lightSwitch)[0] < -.7 / 2:
        p.changeVisualShape(kitchen, lightLink, rgbaColor=[2, 2, 2, 1])
        p.setJointMotorControl2(kitchen, lightBlock, p.POSITION_CONTROL, targetPosition=-.05 / 2)
    elif p.getJointState(kitchen, lightSwitch)[0] >= -.7 / 2 and p.getJointState(kitchen, lightSwitch)[0] < 0:
        p.changeVisualShape(kitchen, lightLink, rgbaColor=[.1, .1, .1, 1])
        p.setJointMotorControl2(kitchen, lightBlock, p.POSITION_CONTROL, targetPosition=0)
