import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2


import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)

PWD = os.path.dirname(os.path.realpath(__name__))

POINT_BIN_PATH = "/mnt/d/bairui/brfile/dataset/rellis_data/Rellis-3D/00000/os1_cloud_node_kitti_bin/002000.bin"
# POINT_BIN_PATH = "/mnt/d/bairui/brfile/dataset/rellis_data/Rellis-3D/00000/vel_cloud_node_kitti_bin/001000.bin"
IMAGE_FILE_PATH = "/mnt/d/bairui/brfile/dataset/rellis_data/Rellis-3D/00000/pylon_camera_node/002000.jpg"
# CALIB_PATH = "/mnt/d/bairui/brfile/dataset/rellis_data/Rellis-3D/00000/calib.txt"
CALIB_PATH = "/mnt/d/bairui/brfile/dataset/rellis_data/Rellis-3D/00000/calib_vel2cam.txt"
# POINT_BIN_PATH = "/mnt/d/bairui/brfile/dataset/SemanticKitti/dataset/sequences/00/velodyne/000001.bin"
# IMAGE_FILE_PATH = "/mnt/d/bairui/brfile/dataset/SemanticKitti/dataset/sequences/00/image_2/000001.png"
# CALIB_PATH = "/mnt/d/bairui/brfile/dataset/SemanticKitti/dataset/sequences/00/calib.txt


def read_calib(calib_path):
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                break
            key, value = line.split(':', 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    calib_out['P0'] = calib_all['P0'].reshape(3, 4)  # 3x4 projection matrix for left camera
    # calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
    logging.info(calib_out['P0'])
    # logging.info(calib_out['P2'])
    calib_out['Tr'] = np.identity(4)  # 4x4 matrix
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
    logging.info( calib_out['Tr'])

    return calib_out


def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind



if __name__ == "__main__":
    points_raw_data = np.fromfile(POINT_BIN_PATH, dtype=np.float32).reshape((-1, 4))
    logging.info("points: {}".format(points_raw_data))
    origin_len = len(points_raw_data)
    points = points_raw_data[:, :3]
    image = Image.open(IMAGE_FILE_PATH)
    image_cv = cv2.imread(IMAGE_FILE_PATH)
    calib = read_calib(CALIB_PATH)
    proj_matrix = np.matmul(calib["P0"], calib["Tr"])
    # proj_matrix = np.matmul(calib["P2"], calib["Tr"])
    logging.info("proj_matric: {}".format(proj_matrix))
    logging.info("points: {}\n       points.shape: {}".format(points, points.shape))
    # logging.info(points.)
    mask_x = np.logical_and(points[:, 0] > -50, points[:, 0] < 50)
    mask_y = np.logical_and(points[:, 1] > -50, points[:, 1] < 50)
    mask_z = np.logical_and(points[:, 2] > -4, points[:, 2] < 2)
    mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
    points1 = points[mask]
    logging.info("points1: {}\n       points.shape: {}".format(points1, points1.shape))
    keep_idx = points1[:, 0] > 0
    print(keep_idx.sum())
    points_hcoords = np.concatenate([points1[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
    logging.info(points_hcoords.shape)
    logging.info(points_hcoords)
    img_points = (proj_matrix @ points_hcoords.T).T
    logging.info(img_points.shape)
    logging.info(img_points)
    print()
    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
    logging.info(img_points)
    logging.info(img_points.shape)
    logging.info(np.max(img_points[:,1]))
    logging.info(np.min(img_points[:,1]))
    logging.info(img_points[:, 0])
    logging.info(img_points[:, 1])
    plt.scatter(img_points[:, 0], image.size[1] - img_points[:, 1])
    plt.show()
    keep_idx_img_pts = select_points_in_frustum(img_points, 0, 0, *image.size)
    logging.info("img size: {}".format(image.size))
    keep_idx[keep_idx] = keep_idx_img_pts
    points_img = img_points[keep_idx_img_pts]
    plt.scatter(points_img[:, 0], image.size[1] - points_img[:, 1])   # image 的坐标原点位于图像右上角，朝右为u，朝下为v
    for point_tmp in points_img:
        cv2.circle(image_cv, (int(point_tmp[0]), int(point_tmp[1])), 1, (255, 0, 0))
    cv2.circle(image_cv, (100, 300), 10, (0, 0, 255))
    cv2.circle(image_cv, (100, 301), 10, (0, 0, 255))
    cv2.circle(image_cv, (101, 300), 10, (0, 0, 255))
    cv2.imwrite(PWD + "/test.png", image_cv)
    plt.show()