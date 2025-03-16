import os
import sys
import numpy as np
import json
import argparse
import subprocess
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description="")
parser.add_argument("--openMVG_bin_dir", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--scene_list", type=str, default=None)
args = parser.parse_args(sys.argv[1:])

openMVG_bin_dir = args.openMVG_bin_dir
dataset_dir = args.dataset_dir
scene_list_path = args.scene_list
ptr_wrapper_id = 2147483649
polymorphic_id = 1073741824

num_cpu_threads = 12

scene_list = []
with open(scene_list_path, 'r') as fin:
    for line in fin:
        scene_list.append(line.strip())

for scene in scene_list:
    openMVG_output_dir = os.path.join(dataset_dir, scene, "openMVG")
    os.makedirs(openMVG_output_dir, exist_ok=True)

    openMVG_matches_dir = os.path.join(openMVG_output_dir, "matches")
    openMVG_reconstruction_dir = os.path.join(openMVG_output_dir, "reconstruction")
    subprocess.run(['mkdir', '-p', openMVG_matches_dir])
    subprocess.run(['mkdir', '-p', openMVG_reconstruction_dir])

    openMVG_init_path = os.path.join(openMVG_matches_dir, "data_openmvg_init.json")

    # Gather scene information
    with open(os.path.join(dataset_dir, scene, "transform.json"), 'r') as pose_fin:
        frames_file = json.load(pose_fin)
    frames = frames_file["frames"]
    img_width = frames_file["width"]
    img_height = frames_file["height"]
    img_dir = os.path.join(dataset_dir, scene, "images")

    json_views = []
    json_extrs = []

    frame_idx = 0
    for frame in frames:
        file_name = frame["file_path"]
        img_idx = os.path.splitext(file_name)[0]

        json_view = {
                    "key" : frame_idx,
                    "value": {
                        "polymorphic_id": polymorphic_id,
                        "ptr_wrapper": {
                            "id": (ptr_wrapper_id + frame_idx),
                            "data": {
                                "local_path": "",
                                "filename" : file_name,
                                "width": img_width,
                                "height": img_height,
                                "id_view": frame_idx,
                                "id_intrinsic": 0,
                                "id_pose": frame_idx
                            }
                        }
                    }
                }

        Twc = np.array(frame["transform_matrix"])
        Twc[1:3, :] *= -1 # nerf style to slam style
        Rwc = Twc[:3,:3]
        twc = Twc[:3,3]
        Rcw = np.linalg.inv(Rwc)
        tcw = -Rcw @ twc
        ccw = twc

        rotation_matrix = Rcw.tolist()
        center = ccw.tolist()

        json_extr = {
            "key" : frame_idx,
            "value": {
                "rotation": rotation_matrix,
                "center": center
            }
        }
        json_views.append(json_view)
        json_extrs.append(json_extr)
        frame_idx = frame_idx + 1

    json_intrs = []
    json_intrs.append({
        "key" : 0,
        "value": {
                "polymorphic_id": ptr_wrapper_id,
                "polymorphic_name": "spherical",
                "ptr_wrapper": {
                    "id": (ptr_wrapper_id + frame_idx),
                    "data": {
                        "value0": {
                            "width": img_width,
                            "height": img_height
                        }
                    }
                }
            }
        })

    json_content = {
        "sfm_data_version": "0.3",
        "root_path" : img_dir,
        "views" : json_views,
        "intrinsics": json_intrs,
        "extrinsics": json_extrs,
        "structure": [],
        "control_points": []
    }

    with open(openMVG_init_path, 'w') as fout:
        json.dump(json_content, fout)

    # Compute sparse point cloud
    ## Features and matches
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ComputeFeatures"),
                    '-i', openMVG_init_path,
                    '-o', openMVG_matches_dir,
                    '-m', "SIFT",
                    '-p', "ULTRA",
                    '-n', num_cpu_threads.__str__()])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_PairGenerator"),
                    '-i', openMVG_init_path,
                    '-o', os.path.join(openMVG_matches_dir, "pairs.bin")])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ComputeMatches"),
                    '-i', openMVG_init_path,
                    '-p', os.path.join(openMVG_matches_dir, "pairs.bin"),
                    '-o', os.path.join(openMVG_matches_dir, 'matches.putative.bin')])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_GeometricFilter"),
                    '-i', openMVG_init_path,
                    '-m', os.path.join(openMVG_matches_dir, 'matches.putative.bin'),
                    '-g', "a",
                    '-o', os.path.join(openMVG_matches_dir, 'matches.f.bin')])

    ## Reconstruction
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_SfM"),
                    '-s', "INCREMENTAL",
                    '-i', openMVG_init_path,
                    '-m', openMVG_matches_dir,
                    '-o', openMVG_reconstruction_dir])


    ## Poses
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    '-i', openMVG_init_path,
                    '-o', os.path.join(openMVG_reconstruction_dir, 'poses_original.ply')])

    recons_sfm_bin_path = os.path.join(openMVG_reconstruction_dir, "sfm_data.bin")
    recons_frames_json_path = os.path.join(openMVG_output_dir, "data_openmvg.json")
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    '-i', recons_sfm_bin_path,
                    '-o', recons_frames_json_path,
                    '-V', '-I', '-E'])
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    '-i', recons_sfm_bin_path,
                    '-o', os.path.join(openMVG_reconstruction_dir, "poses_new.ply"),
                    '-V', '-I', '-E'])

    ## Sparse point cloud
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    'binary',
                    '-i', recons_sfm_bin_path,
                    '-o', os.path.join(openMVG_reconstruction_dir, "point_cloud.bin"),
                    '-V', '-I', '-S'])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ComputeSfM_DataColor"),
                    '-i', os.path.join(openMVG_reconstruction_dir, "point_cloud.bin"),
                    '-o', os.path.join(openMVG_reconstruction_dir, "colorized.ply")])


    # Split into the train and test sets
    train_indices = []
    with open(os.path.join(dataset_dir, scene, "train.txt"), 'r') as train_fin:
        for line in train_fin:
            train_indices.append(line.strip())
    test_indices = []
    with open(os.path.join(dataset_dir, scene, "test.txt"), 'r') as test_fin:
        for line in test_fin:
            test_indices.append(line.strip())

    with open(recons_frames_json_path, 'r') as recons_frames_fin:
        recons_frames_json = json.load(recons_frames_fin)
    recons_views = recons_frames_json["views"]
    recons_extrs = recons_frames_json["extrinsics"]

    extrs = {}
    for extr in recons_extrs:
        extrs[extr["key"]] = extr["value"]

    train_json_views = []
    train_json_extrs = []
    test_json_views = []
    test_json_extrs = []
    train_frame_idx = 0
    test_frame_idx = 0
    for view in recons_views:
        file_name = view["value"]["ptr_wrapper"]["data"]["filename"]
        img_idx = os.path.splitext(file_name)[0]
        if img_idx in train_indices:
            json_view = {
                    "key" : train_frame_idx,
                    "value": {
                        "polymorphic_id": polymorphic_id,
                        "ptr_wrapper": {
                            "id": (ptr_wrapper_id + train_frame_idx),
                            "data": {
                                "local_path": "",
                                "filename" : file_name,
                                "width": img_width,
                                "height": img_height,
                                "id_view": train_frame_idx,
                                "id_intrinsic": 0,
                                "id_pose": train_frame_idx
                            }
                        }
                    }
                }
            json_extr = {
                "key" : train_frame_idx,
                "value": extrs[view["value"]["ptr_wrapper"]["data"]["id_pose"]]
            }
            train_json_views.append(json_view)
            train_json_extrs.append(json_extr)
            train_frame_idx = train_frame_idx + 1

        elif img_idx in test_indices:
            json_view = {
                    "key" : test_frame_idx,
                    "value": {
                        "polymorphic_id": polymorphic_id,
                        "ptr_wrapper": {
                            "id": (ptr_wrapper_id + test_frame_idx),
                            "data": {
                                "local_path": "",
                                "filename" : file_name,
                                "width": img_width,
                                "height": img_height,
                                "id_view": test_frame_idx,
                                "id_intrinsic": 0,
                                "id_pose": test_frame_idx
                            }
                        }
                    }
                }
            json_extr = {
                "key" : test_frame_idx,
                "value": extrs[view["value"]["ptr_wrapper"]["data"]["id_pose"]]
            }
            test_json_views.append(json_view)
            test_json_extrs.append(json_extr)
            test_frame_idx = test_frame_idx + 1

    train_json_intrs = []
    train_json_intrs.append({
        "key" : 0,
        "value": {
                "polymorphic_id": ptr_wrapper_id,
                "polymorphic_name": "spherical",
                "ptr_wrapper": {
                    "id": (ptr_wrapper_id + train_frame_idx),
                    "data": {
                        "value0": {
                            "width": img_width,
                            "height": img_height
                        }
                    }
                }
            }
        })

    test_json_intrs = []
    test_json_intrs.append({
        "key" : 0,
        "value": {
                "polymorphic_id": ptr_wrapper_id,
                "polymorphic_name": "spherical",
                "ptr_wrapper": {
                    "id": (ptr_wrapper_id + test_frame_idx),
                    "data": {
                        "value0": {
                            "width": img_width,
                            "height": img_height
                        }
                    }
                }
            }
        })

    train_json_content = {
        "sfm_data_version": "0.3",
        "root_path" : img_dir,
        "views" : train_json_views,
        "intrinsics": train_json_intrs,
        "extrinsics": train_json_extrs,
        "structure": [],
        "control_points": []
    }
    test_json_content = {
        "sfm_data_version": "0.3",
        "root_path" : img_dir,
        "views" : test_json_views,
        "intrinsics": test_json_intrs,
        "extrinsics": test_json_extrs,
        "structure": [],
        "control_points": []
    }

    train_json_path = os.path.join(openMVG_output_dir, "data_openmvg_train.json")
    test_json_path = os.path.join(openMVG_output_dir, "data_openmvg_test.json")
    with open(train_json_path, 'w') as fout:
        json.dump(train_json_content, fout)
    with open(test_json_path, 'w') as fout:
        json.dump(test_json_content, fout)