import pybullet as p
try:
    from . import config
except ImportError:
    import config

class Robot:

    def __init__(self):
        self.baseStartPosition = config.baseStartPosition
        self.baseStartOrientationQ = p.getQuaternionFromEuler(config.baseStartOrientationE)
        self.jointStartPositions = config.jointStartPositions

        self.id = p.loadURDF("franka_panda/panda.urdf", self.baseStartPosition, self.baseStartOrientationQ, useFixedBase=True)

        i = 0
        for j in range(p.getNumJoints(self.id)):
            jointType = p.getJointInfo(self.id, j)[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                p.resetJointState(self.id, j, self.jointStartPositions[i])
                i += 1
