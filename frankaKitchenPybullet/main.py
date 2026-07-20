import pybullet as p
import pybullet_data
from env import Environment
from robot import Robot

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")

env = Environment()
env.load()

robot = Robot()

while True:
    env.update()
