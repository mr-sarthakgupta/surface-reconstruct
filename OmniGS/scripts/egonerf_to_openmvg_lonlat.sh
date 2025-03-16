#!/bin/bash

openMVG_bin_dir=/home/rapidlab/libs/openMVG/bin

dataset_dir=/home/rapidlab/dataset/omnidirectional/EgoNeRF/OmniBlender
python3 ./egonerf_to_openmvg_omniblender.py --openMVG_bin_dir $openMVG_bin_dir --dataset_dir $dataset_dir --scene_list ./egonerf_scene_list_omniblender.txt

dataset_dir=/home/rapidlab/dataset/omnidirectional/EgoNeRF/Ricoh360
python3 ./egonerf_to_openmvg_train_ricoh360.py --openMVG_bin_dir $openMVG_bin_dir --dataset_dir $dataset_dir --scene_list ./egonerf_scene_list_ricoh360.txt
python3 ./egonerf_to_openmvg_test_ricoh360.py --openMVG_bin_dir $openMVG_bin_dir --dataset_dir $dataset_dir --scene_list ./egonerf_scene_list_ricoh360.txt
