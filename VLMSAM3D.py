import csv
import os
import warnings

from peft import PeftModel
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2
import numpy as np
import open3d as o3d
import torch
import multiprocessing as mp
#import pointops
import random
import argparse
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from os.path import join
from util import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import faiss
warnings.filterwarnings("ignore", category=UserWarning, message="The default value of the antialias parameter")

def get_args():
    parser = argparse.ArgumentParser(description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str, default='2D_data',help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='processed', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, default='result', help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='checkpoints', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05, help='search for closest point')
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"],
                        help="Precision for training")
    parser.add_argument("--dataset", type=str, default="scanrefer", choices=["nr3d", "sr3d", "scanrefer"],
                        help="Dataset to test: nr3d, sr3d, or scanrefer")
    args = parser.parse_args()
    return args

def get_sam(color_image, sam2_predictor, points, bbox):
    """
    SAM2 for segmentation（based on point and bbox）
    """
    # to RGB
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(image_rgb)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=points,
            point_labels=[1, 1],
            box=bbox,
            multimask_output=False
        )
    best_mask = (masks[0] > 0.5)

    group_ids = np.full((color_image.shape[0], color_image.shape[1]), -1, dtype=int)
    group_ids[best_mask] = 0  # bool operation. mask area set ID 0

    return group_ids

def get_2Dmask(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path,reasoning_model
            ,processor,descriptions,object_ids,data_dict):

    # save_2dmask_path = join(save_2dmask_path, scene_name)
    # if not os.path.exists(save_2dmask_path):
    #     os.makedirs(save_2dmask_path)

    #intrinsic_path = join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
    intrinsic_path = join(rgb_path, scene_name, 'intrinsics_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = cv2.imread(depth, -1) # depth both 640,480
    valid_mask = (depth_img != 0)
    color_image_origin = cv2.imread(color)
    color_image = cv2.resize(color_image_origin, (640, 480))

    pose = np.loadtxt(pose)
    #1. 3D coord
    depth_shift = 1000.0
    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

    #2.  intrinsic
    #intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]

    #3. convert to 3D coord
    n = uv_depth.shape[0]
    points = np.ones((n, 4))
    X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 0] = X
    points[:, 1] = Y
    points[:, 2] = uv_depth[:, 2]
    points_world = np.dot(points, np.transpose(pose))

    #4. find corresponding index for 3D point in current frame

    dimension = 3  # 3D
    res = faiss.StandardGpuResources()      # GPU
    cpu_index = faiss.IndexFlatL2(dimension)  # CPU index， L2
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # to GPU 0
    scene_coords_np = np.ascontiguousarray(data_dict[:,:3].astype('float32'))
    gpu_index.add(scene_coords_np)
    points_world_np = np.ascontiguousarray(points_world[:, :3].astype('float32'))
    distances, min_indices_np = gpu_index.search(points_world_np, 1)
    point_instance_ids = data_dict[:,9][min_indices_np.flatten()]


    has_target_list = []
    for object_id in object_ids:
        # whether current frame has objectID
        has_target = np.any(point_instance_ids == object_id)
        has_target_list.append(has_target)

    #5. if no target，return -1
    results = []
    for i, (object_id, description) in enumerate(zip(object_ids, descriptions)):
        group_ids = np.full((color_image.shape[0], color_image.shape[1]), -1, dtype=int)

        color_image_flat = np.reshape(color_image[valid_mask], [-1, 3])
        group_3d = group_ids[valid_mask]
        colors = np.zeros_like(color_image_flat)
        colors[:, 0] = color_image_flat[:, 2]
        colors[:, 1] = color_image_flat[:, 1]
        colors[:, 2] = color_image_flat[:, 0]

        results.append(dict(
            coord=points_world[:, :3],
            color=colors,
            group=group_3d,
            object_id=object_id,
            description=description
        ))

    """
    1.only target object for image and description preprocessing
    2.qwen2.5 vl for point + bbox
    3.get_sam, using sam2 to get mask
    """
    image_qwen = Image.open(color) #.convert('RGB')
    resize_sizex,resize_sizey =  640,480  #depth figure size

    QUESTION_TEMPLATE = \
        "Find '{Question}'." \
        "The image size is 640*480, locate the most closely matched one."\
        "Directly output the answer with one bbox and two points inside the interested object in <answer> </answer> tags. i.e., <answer>{Answer}</answer> in JSON format." \

    target_indices = [i for i, has_target in enumerate(has_target_list) if has_target]
    if not target_indices:
        return results
    batch_size = 15
    all_output_texts = []

    for batch_start in range(0, len(target_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(target_indices))
        batch_indices = target_indices[batch_start:batch_end]

        batch_messages = []
        for i in batch_indices:
            description = descriptions[i]
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_qwen.resize((resize_sizex, resize_sizey), Image.BILINEAR)
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=description.lower().strip("."),
                                                         Answer='{"bbox": [10,100,200,210], "points_1": [30,110], "points_2": [35,180]}')
                    }
                ]
            }]
            batch_messages.append(message)

        batch_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                      batch_messages]

        batch_image_inputs, batch_video_inputs = process_vision_info(batch_messages)

        batch_input_dict = processor(
            text=batch_text,
            images=batch_image_inputs,
            videos=batch_video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            batch_generated_ids = reasoning_model.generate(
                input_ids=batch_input_dict["input_ids"],
                attention_mask=batch_input_dict["attention_mask"],
                pixel_values=batch_input_dict["pixel_values"],
                image_grid_thw=batch_input_dict["image_grid_thw"],
                use_cache=True,
                max_new_tokens=100,
                do_sample=False
            )

        batch_generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_input_dict["input_ids"], batch_generated_ids)
        ]
        batch_output_texts = processor.batch_decode(
            batch_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        all_output_texts.extend(batch_output_texts)

        del batch_input_dict, batch_generated_ids, batch_generated_ids_trimmed
        torch.cuda.empty_cache()

    output_texts = all_output_texts


    for i, output_text in zip(target_indices, output_texts):
        bbox, points = extract_bbox_points(output_text, 1, 1)

        if (points is None) or (bbox is None):
            continue
        else:
            group_ids = get_sam(color_image, mask_generator, points, bbox)

            color_image_flat = np.reshape(color_image[valid_mask], [-1, 3])
            group_3d = group_ids[valid_mask]
            colors = np.zeros_like(color_image_flat)
            colors[:, 0] = color_image_flat[:, 2]
            colors[:, 1] = color_image_flat[:, 1]
            colors[:, 2] = color_image_flat[:, 0]

            results[i] = dict(
                coord=points_world[:, :3],
                color=colors,
                group=group_3d,
                object_id=object_ids[i],
                description=descriptions[i]
            )
            # img = Image.fromarray(group_ids.astype(np.int16), mode='I;16')
            # img.save(join(save_2dmask_path, f"{color_name[0:-4]}_desc{i}.png"))
    return results


def calculate_mask_distance(pcd1, pcd2):

    points1 = pcd1["coord"][pcd1["group"] == 0]
    points2 = pcd2["coord"][pcd2["group"] == 0]

    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)

    return np.linalg.norm(center1 - center2)


def merge_selected_frames(selected_pcds):
    all_coord = np.concatenate([pcd["coord"] for pcd in selected_pcds], axis=0)
    all_color = np.concatenate([pcd["color"] for pcd in selected_pcds], axis=0)
    all_group = np.concatenate([pcd["group"] for pcd in selected_pcds], axis=0)

    merged_dict = dict(coord=all_coord, color=all_color, group=all_group)
    return voxelize(merged_dict)


def full_scene(scene_name, args, rgb_path, mask_save_path, mask_generator, reasoning_model,
            voxelize, processor, scanrefer,lang):

    print(scene_name, flush=True)
    if os.path.exists(join(mask_save_path, scene_name + ".pth")):
        return

    refer_idxs = lang[scene_name]['idx']
    lang_objID,descriptions= [],[]

    for i in refer_idxs:
        object_id = scanrefer[i]['object_id']
        lang_objID.append(int(object_id))
        description = scanrefer[i]['description']
        descriptions.append(description)
    color_dir = join(rgb_path, scene_name, 'color')
    color_names = sorted(
        os.listdir(color_dir),
        key=lambda x: int(x.split('.')[0])
    )

    scene_path = join( args.data_path, "scans", scene_name, "pc_infos.npy" )
    data_dict = np.load(scene_path)

    all_image_descriptions_list = []
    for color_name in color_names:
        pcd_dict = get_2Dmask(scene_name, color_name, rgb_path, mask_generator, mask_save_path,
                           reasoning_model,processor,descriptions,lang_objID,
                            data_dict = data_dict
                            )
        pcd_dict = [voxelize(one_description) for one_description in pcd_dict]
        all_image_descriptions_list.append(pcd_dict)

    all_image_descriptions_list_transposed = list(zip(*all_image_descriptions_list))
    scene_iou_list = []  #for one des, merge all img
    for i, pcd_list in enumerate(all_image_descriptions_list_transposed):
        filtered_pcds = [
            pcd for pcd in pcd_list
            if np.sum(pcd["group"] == 0) > 0
        ]
        if not filtered_pcds:
            scene_iou_list.append(0.0)
            continue
        n_frames = len(filtered_pcds)
        score_matrix = np.zeros((n_frames, n_frames))

        for a in range(n_frames):
            for b in range(a + 1, n_frames):  #
                score = calculate_mask_distance(filtered_pcds[a], filtered_pcds[b])
                score_matrix[a][b] = score
                score_matrix[b][a] = score  #
        frame_distances = np.mean(score_matrix, axis=1)
        if len(frame_distances) == 0:
            scene_iou_list.append(0.0)
            continue
        threshold = np.percentile(frame_distances, 40)
        selected_indices = np.where(frame_distances <= threshold)[0]
        selected_pcds = [filtered_pcds[idx] for idx in selected_indices]

        # 4. merge
        seg_dict = merge_selected_frames(selected_pcds)
        #seg_dict = filter_outliers_cluster(seg_dict)
        #scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()


        scene_coord = torch.tensor(data_dict[:,:3]).cuda().contiguous()
        gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
        gen_group = seg_dict["group"]

        gen_pcd = o3d.geometry.PointCloud()
        gen_pcd.points = o3d.utility.Vector3dVector(gen_coord.cpu().numpy())
        kdtree = o3d.geometry.KDTreeFlann(gen_pcd)

        scene_np = scene_coord.cpu().numpy()
        indices = []
        dis_list = []

        for point in scene_np:
            _, idx, dis = kdtree.search_knn_vector_3d(point, 1)
            indices.append(idx[0])
            dis_list.append(np.sqrt(dis[0]))

        indices = np.array(indices).reshape(-1, 1)
        dis = np.array(dis_list).reshape(-1, 1)

        group = gen_group[indices.reshape(-1)].astype(np.int16)
        mask_dis = dis.reshape(-1) > 0.6
        group[mask_dis] = -1

        group = group.astype(np.int16)

        pred = group
        pred_binary = (pred == 0)

        #point_id = data_dict["instance_gt"]
        point_id = data_dict[:,9]

        object_id = lang_objID[i]
        point_gt = (point_id == object_id)

        intersection = np.sum(pred_binary & point_gt)
        union = np.sum(pred_binary | point_gt)
        current_iou = intersection / (union + 1e-5)
        print(scene_name,i,current_iou)
        scene_iou_list.append(current_iou)

    if len(scene_iou_list) == 0:
        scene_avg_iou = 0.0
    else:
        scene_avg_iou = np.mean(scene_iou_list)

    scene_iou_list_np = np.array(scene_iou_list)
    Precision_25_scene = (scene_iou_list_np > 0.25).sum().astype(float)/scene_iou_list_np.shape[0]

    print(f"\n=== Scene {scene_name} Processing Done ===")
    print(f"Scene average IOU: {scene_avg_iou:.4f}\n", flush=True)
    print(f"Precision_25_scene: {Precision_25_scene:.4f}\n", flush=True)

    return scene_avg_iou,scene_iou_list

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    refer_data = load_dataset_annotations(args)
    lang = build_lang_dict(refer_data, args.dataset)
    scene_names = get_scene_list(refer_data, args.dataset)

    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))


    processor = AutoProcessor.from_pretrained("checkpoints/qwen2.5_3B",
                                              use_fast=False,
                                              padding_side="left")
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "checkpoints/qwen2.5_3B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    #reasoning_model = PeftModel.from_pretrained(reasoning_model, "checkpoints/checkpoint-16994")
    # scene_list = [line.rstrip('\n') for line in open("ScanRefer/ScanRefer_filtered_val.txt", 'r') if line.rstrip('\n')]
    # scene_names = sorted(scene_list)
    #scene_names =["scene0011_00"]#

    all_scene_avg_iou_list=[]
    total_ious =[]
    for scene_name in scene_names:
        scene_avg_iou,ious =full_scene(scene_name, args, args.rgb_path, args.save_2dmask_path, segmentation_model,reasoning_model,
                voxelize, processor, refer_data, lang)

        all_scene_avg_iou_list.append(scene_avg_iou)
        total_ious.extend(ious)
        average_iou_now = np.mean(all_scene_avg_iou_list)
        total_ious_np = np.array(total_ious)
        Precision_25_now = (total_ious_np > 0.25).sum().astype(float) / total_ious_np.shape[0]
        Precision_50_now = (total_ious_np > 0.5).sum().astype(float) / total_ious_np.shape[0]

        print(f"average_iou_now: {average_iou_now:.4f}\n", flush=True)
        print(f"Precision_25_now: {Precision_25_now:.4f}\n", flush=True)
        print(f"Precision_50_now: {Precision_50_now:.4f}\n", flush=True)
    all_scene_avg_iou = np.mean(all_scene_avg_iou_list)
    print(f"all_scene_avg_iou: {all_scene_avg_iou:.4f}\n", flush=True)
    total_ious_all_np = np.array(total_ious)
    Precision_25 = (total_ious_all_np > 0.25).sum().astype(float)/total_ious_all_np.shape[0]
    Precision_50 = (total_ious_all_np > 0.5).sum().astype(float)/total_ious_all_np.shape[0]
    print(f"Precision_25: {Precision_25:.4f}\n", flush=True)
    print(f"Precision_50: {Precision_50:.4f}\n", flush=True)



