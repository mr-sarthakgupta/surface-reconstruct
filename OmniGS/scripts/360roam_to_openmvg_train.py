import os
import sys
import numpy as np
import json
import argparse
import subprocess

parser = argparse.ArgumentParser(description="")
parser.add_argument("--openMVG_bin_dir", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--scene_list", type=str, default=None)
parser.add_argument("--img_width", type=int, default=None)
parser.add_argument("--img_height", type=int, default=None)
args = parser.parse_args(sys.argv[1:])

openMVG_bin_dir = args.openMVG_bin_dir
dataset_dir = args.dataset_dir
scene_list_path = args.scene_list
img_width = args.img_width
img_height = args.img_height
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
    openMVG_init_path = os.path.join(openMVG_output_dir, "data_openmvg.json")

    # Gather scene information
    with open(os.path.join(dataset_dir, scene, "pose_c2w.json"), 'r') as pose_fin:
        frames_file = json.load(pose_fin)
    frames = frames_file["train"]
    img_dir = os.path.join(dataset_dir, scene, "images")

    json_views = []
    json_extrs = []

    frame_idx = 0
    for frame in frames:
        json_view = {
                    "key" : frame_idx,
                    "value": {
                        "polymorphic_id": polymorphic_id,
                        "ptr_wrapper": {
                            "id": (ptr_wrapper_id + frame_idx),
                            "data": {
                                "local_path": "",
                                "filename" : frame["rgb_file"],
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

    # Compute sparse point cloud with known poses
    openMVG_matches_dir = os.path.join(openMVG_output_dir, "matches")
    openMVG_reconstruction_dir = os.path.join(openMVG_output_dir, "reconstruction")
    subprocess.run(['mkdir', '-p', openMVG_matches_dir])
    subprocess.run(['mkdir', '-p', openMVG_reconstruction_dir])

    ## Features and matches
    sfm_data_json_path = os.path.join(openMVG_matches_dir, 'sfm_data.json')
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    '-i', openMVG_init_path,
                    '-o', sfm_data_json_path])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ComputeFeatures"),
                    '-i', sfm_data_json_path,
                    '-o', openMVG_matches_dir,
                    '-m', "SIFT",
                    '-n', num_cpu_threads.__str__()])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_PairGenerator"),
                    '-i', sfm_data_json_path,
                    '-o', os.path.join(openMVG_matches_dir, "pairs.bin")])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ComputeMatches"),
                    '-i', sfm_data_json_path,
                    '-p', os.path.join(openMVG_matches_dir, "pairs.bin"),
                    '-o', os.path.join(openMVG_matches_dir, 'matches.putative.bin')])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_GeometricFilter"),
                    '-i', sfm_data_json_path,
                    '-m', os.path.join(openMVG_matches_dir, 'matches.putative.bin'),
                    '-g', "a",
                    '-o', os.path.join(openMVG_matches_dir, 'matches.f.bin')])

    ## Reconstruction
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ComputeStructureFromKnownPoses"),
                    '-i', sfm_data_json_path,
                    '-m', openMVG_matches_dir,
                    '-p', os.path.join(openMVG_matches_dir, "pairs.bin"),
                    '-o', os.path.join(openMVG_reconstruction_dir, "sfm_data.bin")])

    ## Poses
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    '-i', sfm_data_json_path,
                    '-o', os.path.join(openMVG_reconstruction_dir, 'poses_original.ply')])

    recons_sfm_bin_path = os.path.join(openMVG_reconstruction_dir, "sfm_data.bin")
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    'binary',
                    '-i', recons_sfm_bin_path,
                    '-o', os.path.join(openMVG_reconstruction_dir, "poses.json"),
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

    ## Convert panoramic images to cubic perspective ones
    cubic_dir = os.path.join(openMVG_reconstruction_dir, "cubic")
    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_openMVGSpherical2Cubic"),
                    '-i', recons_sfm_bin_path,
                    '-o', cubic_dir])

    subprocess.run([os.path.join(openMVG_bin_dir, "openMVG_main_ConvertSfM_DataFormat"),
                    '-i', os.path.join(cubic_dir, "sfm_data_perspective.bin"),
                    '-o', os.path.join(cubic_dir, "sfm_data_perspective.json")])


    # Remove the bottom perspective, because the base car is dynamic
    with open(os.path.join(cubic_dir, "sfm_data_perspective.json"), 'r') as perspective_fin:
        perspective_file = json.load(perspective_fin)
    views_in = perspective_file["views"]
    intrs_in = perspective_file["intrinsics"]
    extrs_in = perspective_file["extrinsics"]
    perspective_img_dir = perspective_file["root_path"]

    views_out = []
    intrs_out = intrs_in
    extrs_out = []

    frame_idx = 0
    key_idx = -1
    for view in views_in:
        key_idx = key_idx + 1
        img_filename = view["value"]["ptr_wrapper"]["data"]["filename"]
        perpective_idx = os.path.splitext(img_filename)[0].split('_')[3]
        if perpective_idx == "00000005":
            continue
        else:
            view_out =  {
                    "key" : frame_idx,
                    "value": {
                        "polymorphic_id": polymorphic_id,
                        "ptr_wrapper": {
                            "id": (ptr_wrapper_id + frame_idx),
                            "data": {
                                "local_path": "",
                                "filename" : img_filename,
                                "width": view["value"]["ptr_wrapper"]["data"]["width"],
                                "height": view["value"]["ptr_wrapper"]["data"]["height"],
                                "id_view": frame_idx,
                                "id_intrinsic": 0,
                                "id_pose": frame_idx
                            }
                        }
                    }
                }
            extr = extrs_in[key_idx]
            extr_out = {
                "key" : frame_idx,
                "value": {
                    "rotation": extr["value"]["rotation"],
                    "center": extr["value"]["center"]
                }
            }
            views_out.append(view_out)
            extrs_out.append(extr_out)
            frame_idx = frame_idx + 1

    perspective_content = {
        "sfm_data_version": "0.3",
        "root_path" : perspective_img_dir,
        "views" : views_out,
        "intrinsics": intrs_out,
        "extrinsics": extrs_out,
        "structure": [],
        "control_points": []
    }

    with open(os.path.join(openMVG_output_dir, "data_openmvg_perspective.json"), 'w') as perspective_fout:
        json.dump(perspective_content, perspective_fout)



