import os
import yaml
import numpy as np


import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)




DATA_PATH = "/mnt/d/bairui/brfile/dataset/rellis_data/Rellis-3D/00000"

##### os12cam
transforms_yaml_path = os.path.join(DATA_PATH, "transforms.yaml")
with open(transforms_yaml_path, 'r') as stream1:
    transforms_yaml = yaml.safe_load(stream1)
w = transforms_yaml['os1_cloud_node-pylon_camera_node']['q']['w']
x = transforms_yaml['os1_cloud_node-pylon_camera_node']['q']['x']
y = transforms_yaml['os1_cloud_node-pylon_camera_node']['q']['y']
z = transforms_yaml['os1_cloud_node-pylon_camera_node']['q']['z']

T_matrix = np.identity(4)
T_matrix[0][0] = 1 - (2 * y * y) - (2 * z * z)
T_matrix[0][1] = (2 * x * y) - (2 * w * z)
T_matrix[0][2] = (2 * x * z) + (2 * w * y)
T_matrix[0][3] = transforms_yaml['os1_cloud_node-pylon_camera_node']['t']['x']
T_matrix[1][0] = (2 * x * y) + (2 * w * z)
T_matrix[1][1] = 1 - (2 * x * x) - (2 * z * z)
T_matrix[1][2] = (2 * y * z) - (2 * w * x)
T_matrix[1][3] = transforms_yaml['os1_cloud_node-pylon_camera_node']['t']['y']
T_matrix[2][0] = (2 * x * z) - (2 * w * y)
T_matrix[2][1] = (2 * y * z) + (2 * w * x)
T_matrix[2][2] = 1 - (2 * x * x) - (2 * y * y)
T_matrix[2][3] = transforms_yaml['os1_cloud_node-pylon_camera_node']['t']['z']
logging.info("T_matrix: {}".format(T_matrix))    # Basler Camera to Ouster LiDAR
T_matrix_inv = np.linalg.inv(T_matrix)
logging.info("T_matrix_inv: {}".format(T_matrix_inv))   # os12cam
# tmp = np.matmul(T_matrix, T_matrix_inv)
# logging.info(tmp)
# stop_here()

##### vel2os1 
vel2os1_yaml_path = os.path.join(DATA_PATH, "vel2os1.yaml")
with open(vel2os1_yaml_path, 'r') as stream2:
    vel2os1_yaml = yaml.safe_load(stream2)
w2 = vel2os1_yaml['vel2os1']['q']['w']
x2 = vel2os1_yaml['vel2os1']['q']['x']
y2 = vel2os1_yaml['vel2os1']['q']['y']
z2 = vel2os1_yaml['vel2os1']['q']['z']

T2_matrix = np.identity(4)
T2_matrix[0][0] = 1 - (2 * y2 * y2) - (2 * z2 * z2)
T2_matrix[0][1] = (2 * x2 * y2) - (2 * w2 * z2)
T2_matrix[0][2] = (2 * x2 * z2) + (2 * w2 * y2)
T2_matrix[0][3] = vel2os1_yaml['vel2os1']['t']['x']
T2_matrix[1][0] = (2 * x2 * y2) + (2 * w2 * z2)
T2_matrix[1][1] = 1 - (2 * x2 * x2) - (2 * z2 * z2)
T2_matrix[1][2] = (2 * y2 * z2) - (2 * w2 * x2)
T2_matrix[1][3] = vel2os1_yaml['vel2os1']['t']['y']
T2_matrix[2][0] = (2 * x2 * z2) - (2 * w2 * y2)
T2_matrix[2][1] = (2 * y2 * z2) + (2 * w2 * x2)
T2_matrix[2][2] = 1 - (2 * x2 * x2) - (2 * y2 * y2)
T2_matrix[2][3] = vel2os1_yaml['vel2os1']['t']['z']
logging.info("T2_matrix: {}".format(T2_matrix))

T3_matrix = np.matmul(T_matrix_inv, T2_matrix)
logging.info("T3_matrix: {}".format(T3_matrix))




cam_info_path = os.path.join(DATA_PATH, "camera_info.txt")
with open(cam_info_path, 'r') as stream2:
    for line in stream2.readlines():
        if line == '\n':
            break
        cam_info = np.array([float(x) for x in line.split()])

for x in cam_info:
    logging.info(x)
fx = cam_info[0]
fy = cam_info[1]
cx = cam_info[2]
cy = cam_info[3]
