import csv
from os.path import join
import torch
import open3d as o3d
import os
import copy
from PIL import Image
import json
#import pointops
import re

import numpy as np
from sklearn.cluster import DBSCAN

class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

def extract_bbox_points(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'
    json_match = re.search(json_pattern, output_text)

    content_bbox, points = None, None

    if json_match:
        try:
            #data = json.loads(json_match.group(0))
            json_str = json_match.group(0)
            json_str = re.sub(r'(\d+)\.(?=\D)', r'\1.0', json_str)
            data = json.loads(json_str)

            # bbox
            bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
            if bbox_key and isinstance(data[bbox_key], list) and len(data[bbox_key]) == 4:
                try:
                    content_bbox = data[bbox_key]
                    content_bbox = [round(int(content_bbox[0]) * x_factor), round(int(content_bbox[1]) * y_factor),
                                    round(int(content_bbox[2]) * x_factor), round(int(content_bbox[3]) * y_factor)]
                except (ValueError, TypeError, IndexError):
                    content_bbox = None

            # points
            points_keys = [key for key in data.keys() if 'points' in key.lower()]
            if len(points_keys) >= 2:
                try:
                    point1 = data[points_keys[0]]
                    point2 = data[points_keys[1]]
                    if isinstance(point1, list) and len(point1) == 2 and \
                            isinstance(point2, list) and len(point2) == 2:
                        point1 = [round(int(point1[0]) * x_factor), round(int(point1[1]) * y_factor)]
                        point2 = [round(int(point2[0]) * x_factor), round(int(point2[1]) * y_factor)]
                        points = [point1, point2]
                except (ValueError, TypeError, IndexError):
                    points = None

        except json.JSONDecodeError as e:
            print(f"JSON error: {e}")
            print(f"text: {json_match.group(0)}")
            content_bbox, points = None, None

    default_bbox = [0, 0, 0, 0]
    default_points = [[0, 0], [0, 0]]

    content_bbox = content_bbox if content_bbox is not None else default_bbox
    points = points if points is not None else default_points

    return content_bbox, points


def visualize_point_clouds(input_dict_0, input_dict_1):
    pcd_0 = o3d.geometry.PointCloud()
    pcd_0.points = o3d.utility.Vector3dVector(input_dict_0["coord"])
    pcd_0.colors = o3d.utility.Vector3dVector(input_dict_0["color"] / 255.0)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(input_dict_1["coord"])
    pcd_1.colors = o3d.utility.Vector3dVector(input_dict_1["color"] / 255.0)

    translation = np.array([2.0, 0.0, 0.0])
    pcd_1.translate(translation)

    o3d.visualization.draw_geometries([pcd_0, pcd_1])


def load_nr3d_annotations(data_path):

    annotations = []
    refer_path = join(data_path, 'ReferIt3D/nr3d.csv')
    with open(refer_path) as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        headers = {header: h for h, header in enumerate(headers)}

        for line in csv_reader:
            scan_id = line[headers['scan_id']]

            annotations.append({
                'scene_id': scan_id,
                'object_id': int(line[headers['target_id']]),
                'description': line[headers['utterance']],
                'target': line[headers['instance_type']],
                'dataset': 'nr3d'
            })

    return annotations

def load_sr3d_annotations(data_path):

    annotations = []
    refer_path = join(data_path,'ReferIt3D/sr3d_test.csv')
    with open(refer_path) as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        headers = {header: h for h, header in enumerate(headers)}

        for line in csv_reader:
            scan_id = line[headers['scan_id']]

            # Only include mentions that mention target class
            if str(line[headers['mentions_target_class']]).lower() != 'true':
                continue

            annotations.append({
                'scene_id': scan_id,
                'object_id': int(line[headers['target_id']]),
                'description': line[headers['utterance']],
                'target': line[headers['instance_type']],
                'dataset': 'sr3d'
            })

    return annotations

def get_scene_list(refer_data, dataset_type):
    if dataset_type == "scanrefer":
        scene_list_file = "ScanRefer/ScanRefer_filtered_val.txt"
        return [line.rstrip('\n') for line in open(scene_list_file, 'r') if line.rstrip('\n')]
    else:
        scene_set = set(data['scene_id'] for data in refer_data)
        return sorted(list(scene_set))


def load_dataset_annotations(args):
    if args.dataset == "nr3d":
        return load_nr3d_annotations(args.data_path)
    elif args.dataset == "sr3d":
        return load_sr3d_annotations(args.data_path)
    elif args.dataset == "scanrefer":
        return json.load(open('ScanRefer/ScanRefer_filtered.json'))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_lang_dict(refer_data, dataset_type):
    lang = {}

    for i, data in enumerate(refer_data):
        if dataset_type == "scanrefer":
            scene_id = data['scene_id']
            object_id = data['object_id']
        else:  # nr3d or sr3d
            scene_id = data['scene_id']
            object_id = data['object_id']

        if scene_id not in lang:
            lang[scene_id] = {'idx': []}

        lang[scene_id]['idx'].append(i)

    return lang