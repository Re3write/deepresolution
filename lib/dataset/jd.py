# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import cv2
import json_tricks as json
import numpy as np

from lib.dataset.JointsforFace import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms

logger = logging.getLogger(__name__)


class JDDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23],
                           [10, 22],
                           [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],
                           [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],
                           [55, 65], [56, 64], [57, 63], [58, 62], [59, 61],
                           [66, 79], [67, 78], [68, 77], [69, 76], [70, 75], [71, 82], [72, 81], [73, 80], [74, 83],
                           [84, 90], [85, 89], [86, 88], [91, 95], [92, 94], [96, 100], [97, 99], [101, 103],
                           [104, 105]]

        # deal with class names
        # cats = [cat['name']
        #         for cat in self.coco.loadCats(self.coco.getCatIds())]
        # self.classes = ['__background__'] + cats
        # logger.info('=> classes: {}'.format(self.classes))
        # self.num_classes = len(self.classes)
        # self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        # self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        # self._coco_ind_to_class_ind = dict(
        #     [
        #         (self._class_to_coco_ind[cls], self._class_to_ind[cls])
        #         for cls in self.classes[1:]
        #     ]
        # )

        # load image file names


        self.num_joints = 106
        # self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
        #                    [9, 10], [11, 12], [13, 14], [15, 16]]
        # self.parent_ids = None
        # self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        # self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        # self.joints_weight = np.array(
        #     [
        #         1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
        #         1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
        #     ],
        #     dtype=np.float32
        # ).reshape((self.num_joints, 1))

        self.db = self._get_db()

        # if is_train and cfg.DATASET.SELECT_DATA:
        #     self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        if self.is_train:
            # use ground truth bbox or self.use_gt_bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            gt_db = self.load_val()
        # else:
        #     # use bbox from detection
        #     gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        dataset_dir = "/home/sk49/workspace/dataset/jingdong_full_dataset/picture"
        ann_path = "/home/sk49/workspace/cy/jd/landmark.txt"

        with open(ann_path, 'r') as file_to_read:
            for line in file_to_read.readlines():
                rec = []
                line = line.strip('\n')
                content = line.split(" ")
                img_name = content[0]
                img_path = os.path.join(dataset_dir, content[0])
                i = 1
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)

                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = int(float(content[ipt * 2 + 1]))
                    joints_3d[ipt, 1] = int(float(content[i * 2 + 2]))
                    joints_3d[ipt, 2] = 0
                    t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                img = cv2.imread(img_path)
                h, w, c = img.shape
                center, scale = self._box2cs([0, 0, w, h])
                rec.append({
                    'image': img_path,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                })
                gt_db.extend(rec)

        return gt_db

    def load_val(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        dataset_dir = "/home/sk49/workspace/dataset/jingdong_full_dataset/picture"
        ann_path = "/home/sk49/workspace/cy/jd/val.txt"

        with open(ann_path, 'r') as file_to_read:
            for line in file_to_read.readlines():
                rec = []
                line = line.strip('\n')
                content = line.split(" ")
                img_name = content[0]
                img_path = os.path.join(dataset_dir, content[0])
                i = 1
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)

                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = int(float(content[ipt * 2 + 1]))
                    joints_3d[ipt, 1] = int(float(content[i * 2 + 2]))
                    joints_3d[ipt, 2] = 0
                    t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                img = cv2.imread(img_path)
                h, w, c = img.shape
                center, scale = self._box2cs([0, 0, w, h])
                rec.append({
                    'image': img_path,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                })
                gt_db.extend(rec)

        return gt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],   #/200?
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, data_name, file_name)

        return image_path

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        print('你好')
