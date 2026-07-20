# import pybullet as p
# import time
# import pybullet_data
# import os
# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
# # startPos = [0,0,1]
# # startOrientation = p.getQuaternionFromEuler([0,0,0])
# folder_path="./"
# def get_file_names_in_folder(folder_path):
#             files_list = []
            
#             # Check if the folder path exists
#             if os.path.exists(folder_path) and os.path.isdir(folder_path):
#                 # Iterate through files in the folder and append their names to the list
#                 for file_name in os.listdir(folder_path):
#                     files_list.append(file_name)
#             else:
#                 print("Folder path doesn't exist or is not a directory.")
            
#             return files_list
# file_list=get_file_names_in_folder(folder_path)
# for i in file_list:
#     try:
#         boxId = p.loadURDF("./"+str(i)+"/urdf/"+str(i)+".urdf",[0,0,0],[0,0,0,1],useFixedBase=True)
#     except:
#         pass
# for i in range (10000):
#     p.stepSimulation()
#     time.sleep(1./100.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()


import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(0)
# base plane
planeId = p.loadURDF("./plan/urdf/plan.urdf",[-3,3,3.62],[-1,0,0,-1],useFixedBase =True)
# side wall
sidewallId = p.loadURDF("./side_wall/urdf/side_wall.urdf",[-1.65,3,5.0],[1,-1,0,0],useFixedBase =True)
sidewallId2 = p.loadURDF("./side_wall/urdf/side_wall.urdf",[-3,-5.53,5],[1,0,0,0],useFixedBase =True)
sidewallId3 = p.loadURDF("./side_wall/urdf/side_wall.urdf",[0.6,-2.0,0],[1,1,1,-1],useFixedBase =True)

startPos = [0,0,-0.07]
# startOrientation = p.getQuaternionFromEuler([0,0,0])
startOrientation = [0,0,0,1]   
table = p.loadURDF("./wood_table/urdf/wood_table.urdf",[0,0,-0.42], [-1,0,0,1],useFixedBase =True)

boxId = p.loadURDF("./banana/urdf/banana.urdf",[0.62,1,0.66], startOrientation,useFixedBase =True)
boxId = p.loadURDF("./carrot/urdf/carrot.urdf",[0.75,1.0,0.64], startOrientation,useFixedBase =True)
boxId = p.loadURDF("./strawberry/urdf/strawberry.urdf",[0.57,0.9,0.73], startOrientation,useFixedBase =True)
boxId = p.loadURDF("./cup/urdf/cup.urdf",[0.30,0.9,0.725], [-1,0,0,1],useFixedBase =True)
boxId = p.loadURDF("./plastic_water_bottle/urdf/plastic_water_bottle.urdf",[0.30,0.9,0.70], [-1,0,0,1],useFixedBase =True)
boxId = p.loadURDF("./Syring/urdf/Syring.urdf",[0.90,0.7,0.68], [0,0,1,1],useFixedBase =True)
boxId = p.loadURDF("./syrup bottle/urdf/syrup bottle.urdf",[0.90,0.65,0.72], [-1,0,0,1],useFixedBase =True)
trayId= p.loadURDF("./Tray/urdf/Tray.urdf",[0.20,0.15,0.45], [-1,0,0,1],useFixedBase =True)
rackId= p.loadURDF("./rack/urdf/rack.urdf",[1.4,0.7,0.71], [-1,1,-1,1],useFixedBase =True)
ointment=p.loadURDF("./ointment_candid/urdf/ointment_candid.urdf",[0.90,0.80,0.63], [-1,0,0,1],useFixedBase =True)
nasal_spray=p.loadURDF("./nasal_spray/urdf/nasal_spray.urdf",[1.1,0.95,0.74], [-1,-1,-1,1],useFixedBase =True)
spectacle_foldable=p.loadURDF("./Spectacle_foldable/urdf/Spectacle_foldable.urdf",[0.6,1.2,0.69], [0,0,0,-1],useFixedBase =True)
# franka_robot=p.loadURDF("./franka_emika_panda_pybullet-master/panda_robot/model_description/panda.urdf",[0.62,0.62,0.61], [0,0,1,1],useFixedBase =True)
bucket=p.loadURDF("./bucket/bucket/Bucket.urdf",[0.65,1.2,0.74], [0,0,0,-1],useFixedBase =True)
bucket_cap=p.loadURDF("./bucket/bucket_cap/Bucket_cap.urdf",[0.65,1.2,0.85], [0,0,0,-1])

franka_robot=p.loadURDF("./franka_panda/panda.urdf",[0.62,0.62,0.61], [0,0,1,1],useFixedBase =True)

# Increase mass of bottle
# p.changeDynamics(bottleId, -1, mass=10)
# # Increase friction on bottle surface
# p.changeDynamics(bucket, -1, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)

# p.changeDynamics(bucket_cap, -1, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)

# # Increase friction on the surface of the grippers
# p.changeDynamics(franka_robot, 9, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)
# p.changeDynamics(franka_robot, 10, lateralFriction=1000000000, spinningFriction=1000000000, rollingFriction=1000000000)



table_pos, table_orn = p.getBasePositionAndOrientation(table)
print(table_pos)
# box_size = p.getCollisionShapeData(boxId, -1)[0][3]
# box_placement_pos = [table_pos[0], table_pos[1], table_pos[2] + 0.5 * box_size[2]]
# p.resetBasePositionAndOrientation(boxId, box_placement_pos, table_orn)



# #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./500.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()

