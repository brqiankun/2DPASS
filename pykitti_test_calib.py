import os

import matplotlib
import numpy as np
import pykitti

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

basedir = '/media/br/kitti_odometry/dataset'
sequence = '00'
uncerts = ''
preds = ''
gt = ''
img = ''
lidar = ''
projected_uncert = ''
projected_preds = ''

dataset = pykitti.odometry(basedir, sequence)

EXTENSIONS_LABEL = ['.label']
EXTENSIONS_LIDAR = ['.bin']
EXTENSIONS_IMG = ['.png']


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_lidar(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LIDAR)


def is_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMG)


# def plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name):
velo_points = np.fromfile("/media/br/kitti_odometry/dataset/sequences/00/velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
# try:
#   cam2_image = plt.imread(cam2_image_name)
# except IOError:
#   print('detect error img %s' % label_name)

# plt.imshow(cam2_image)

if True:

  # Project points to camera.
  print(dataset.calib.T_cam2_velo)
  cam2_points = dataset.calib.T_cam2_velo.dot(velo_points.T).T
  print(cam2_points)
  raise RuntimeError
  # Filter out points behind camera
  idx = cam2_points[:, 2] > 0
  print(idx)
  # velo_points_projected = velo_points[idx]
  cam2_points = cam2_points[idx]
  labels_projected = labels[idx]
  uncert_projected = uncerts[idx]

  # Remove homogeneous z.
  cam2_points = cam2_points[:, :3] / cam2_points[:, 2:3]

  # Apply instrinsics.
  intrinsic_cam2 = dataset.calib.K_cam2
  cam2_points = intrinsic_cam2.dot(cam2_points.T).T[:, [1, 0]]
  cam2_points = cam2_points.astype(int)

  # for i in range(0, cam2_points.shape[0]):
  #     u, v = cam2_points[i, :]
  #     label = labels_projected[i]
  #     uncert = uncert_projected[i]
  #     if label > 0 and v > 0 and v < 1241 and u > 0 and u < 376:
  #         uncert_mean[learning_map[label]] += uncert
  #         total_points_per_class[learning_map[label]] += 1
  #         m_circle = plt.Circle((v, u), 1,
  #                               color=matplotlib.cm.viridis(uncert),
  #                               alpha=0.4,
  #                               # color=color_map[label][..., ::-1]
  #                               )
  #         plt.gcf().gca().add_artist(m_circle)

    # plt.axis('off')
    # path = os.path.join(basedir + 'sequences/' + sequence + projected_uncert)
    # plt.savefig(path + label_name.split('/')[-1].split('.')[0] + '.png', bbox_inches='tight', transparent=True,
    #             pad_inches=0)


# with futures.ProcessPoolExecutor() as pool:
# for label_uncert, label_name, lidar_name, cam2_image_name in zip(scan_uncert, scan_preds, dataset.velo_files,
#                                                                  dataset.cam2_files):
#     print(label_name.split('/')[-1])
#     # if label_name == '/SPACE/DATA/SemanticKITTI/dataset/sequences/13/predictions/preds/001032.label':
    # plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name)
# print(total_points_per_class)
# print(uncert_mean)
# if __name__ == "__main__":
    # pass