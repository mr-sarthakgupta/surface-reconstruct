/**
 * This file is part of OmniGS
 *
 * Copyright (C) 2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * OmniGS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OmniGS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with OmniGS.
 * If not, see <http://www.gnu.org/licenses/>.
 * 
 * OmniGS is Derivative Works of Gaussian Splatting.
 * Its usage should not break the terms in LICENSE.md.
 */

#include <unordered_map>
#include <filesystem>
#include <fstream>

#include <torch/torch.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "third_party/colmap/utils/endian.h"
#include "third_party/tinyply/tinyply.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

void readViewsOpenMVG(
    std::shared_ptr<GaussianMapper> pMapper,
    const std::filesystem::path& path)
{
    std::ifstream in_stream;
    in_stream.open(path);
    if (!in_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + path.string());

    Json::Value json_root;
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(builder, in_stream, &json_root, &errs))
        throw std::runtime_error(errs);

    // Cameras
    const std::size_t num_cameras = json_root["intrinsics"].size();
    int i = 1;
    for (const auto& intr : json_root["intrinsics"]) {
        class Camera camera;
        camera.camera_id_ = intr["key"].asUInt();

        std::cout << "                                                                             \r";
        std::cout.flush();
        std::cout << "Loading camera " << camera.camera_id_ << ", " << i << "/" << num_cameras << "\r";
        std::cout.flush();
        i++;

        camera.setModelId(Camera::CameraModelType::LONLAT);
        camera.width_ = intr["value"]["ptr_wrapper"]["data"]["value0"]["width"].asUInt64();
        camera.height_ = intr["value"]["ptr_wrapper"]["data"]["value0"]["height"].asUInt64();

        camera.num_gaus_pyramid_sub_levels_ = pMapper->num_gaus_pyramid_sub_levels_;
        camera.gaus_pyramid_width_.resize(pMapper->num_gaus_pyramid_sub_levels_);
        camera.gaus_pyramid_height_.resize(pMapper->num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < pMapper->num_gaus_pyramid_sub_levels_; ++l) {
            camera.gaus_pyramid_width_[l] = camera.width_ * pMapper->kf_gaus_pyramid_factors_[l];
            camera.gaus_pyramid_height_[l] = camera.height_ * pMapper->kf_gaus_pyramid_factors_[l];
        }

        cv::Mat K = (
            cv::Mat_<float>(3, 3)
                << camera.width_, 0.f, camera.width_ / 2.0f,
                    0.f, camera.height_, camera.height_ / 2.0f,
                    0.f, 0.f, 1.f
        );
        camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, pMapper->isdoingGausPyramidTraining());

        pMapper->undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                camera.undistort_mask, pMapper->device_type_);

        cv::Mat viewer_main_undistort_mask;
        int viewer_image_height_main_ = camera.height_ * pMapper->rendered_image_viewer_scale_main_;
        int viewer_image_width_main_ = camera.width_ * pMapper->rendered_image_viewer_scale_main_;
        cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
        pMapper->viewer_main_undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                viewer_main_undistort_mask, pMapper->device_type_);

        if (!pMapper->viewer_camera_id_set_) {
            pMapper->viewer_camera_id_ = camera.camera_id_;
            pMapper->viewer_camera_id_set_ = true;
        }

        pMapper->scene_->addCamera(camera);
    }
    std::cout << std::endl;

    // Frames
    std::unordered_map<std::size_t, Json::Value> views;
    std::unordered_map<std::size_t, Json::Value> extrinsics;
    // Read json data
    for (const auto& json_view : json_root["views"])
        views.emplace(json_view["key"].asUInt64(), json_view["value"]);
    for (const auto& json_extrinsic : json_root["extrinsics"])
        extrinsics.emplace(json_extrinsic["key"].asUInt64(), json_extrinsic["value"]);
    std::filesystem::path image_dir(json_root["root_path"].asString());
    std::size_t num_frames = views.size();

    i = 1;
    for (const auto& view : views) {
        std::size_t fid = view.first;
        auto json_view = view.second;
        auto json_extrinsic = extrinsics[json_view["ptr_wrapper"]["data"]["id_pose"].asUInt64()];

        std::cout << "                                                                             \r";
        std::cout.flush();
        std::cout << "Loading frame " << fid << ", " << i << "/" << num_frames << "\r";
        std::cout.flush();
        i++;

        std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(fid);
        pkf->img_filename_ = json_view["ptr_wrapper"]["data"]["filename"].asString();
        pkf->zfar_ = pMapper->z_far_;
        pkf->znear_ = pMapper->z_near_;

        // Pose
        Eigen::Vector3f ccw;
        ccw.x() = json_extrinsic["center"][0].asFloat();
        ccw.y() = json_extrinsic["center"][1].asFloat();
        ccw.z() = json_extrinsic["center"][2].asFloat();
        Eigen::Matrix3f Rcw;
        Rcw(0, 0) = json_extrinsic["rotation"][0][0].asFloat();
        Rcw(0, 1) = json_extrinsic["rotation"][0][1].asFloat();
        Rcw(0, 2) = json_extrinsic["rotation"][0][2].asFloat();
        Rcw(1, 0) = json_extrinsic["rotation"][1][0].asFloat();
        Rcw(1, 1) = json_extrinsic["rotation"][1][1].asFloat();
        Rcw(1, 2) = json_extrinsic["rotation"][1][2].asFloat();
        Rcw(2, 0) = json_extrinsic["rotation"][2][0].asFloat();
        Rcw(2, 1) = json_extrinsic["rotation"][2][1].asFloat();
        Rcw(2, 2) = json_extrinsic["rotation"][2][2].asFloat();
        Eigen::Vector3f tcw = -Rcw * ccw;
        pkf->setPose(
            Eigen::Quaterniond(Rcw.cast<double>()),
            tcw.cast<double>());

        try {
            // Camera
            unsigned int camera_id = json_view["ptr_wrapper"]["data"]["id_intrinsic"].asUInt();
            Camera& camera = pMapper->scene_->cameras_.at(camera_id);
            std::size_t width = json_view["ptr_wrapper"]["data"]["width"].asUInt64();
            std::size_t height = json_view["ptr_wrapper"]["data"]["height"].asUInt64();
            assert(width == camera.width_ && height == camera.height_);
            pkf->setCameraParams(camera);
            // Transformations
            pkf->computeTransformTensors();
            // Image
            std::filesystem::path image_path = image_dir / pkf->img_filename_;
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            cv::cvtColor(image, image, CV_BGR2RGB);
            image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
            cv::Mat imgRGB_undistorted;
            camera.undistortImage(image, imgRGB_undistorted);
            pkf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image, pMapper->device_type_);
            // Prepare for gaussian-pyramid-based training
            pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
            pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
            pkf->gaus_pyramid_times_of_use_ = pMapper->kf_gaus_pyramid_times_of_use_;

            pMapper->scene_->addKeyframe(pkf, &pMapper->kfid_shuffled_);
        }
        catch (std::out_of_range) {
            throw std::runtime_error("[GaussianMapper::keyframesFromJson]KeyFrame Camera not found!");
        }
    }
    std::cout << std::endl;
}

void readScene(std::shared_ptr<GaussianMapper> pMapper)
{
    auto& model_params = pMapper->getGaussianModelParams();
    auto scene = pMapper->scene_;

    std::filesystem::path frames_file = model_params.source_path_ / "openMVG/data_openmvg_test.json";

    readViewsOpenMVG(pMapper, frames_file);
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_dataset_directory/"           /*2*/
                  << " path_to_ply_flie_to_be_tested/"       /*3*/
                  << " path_to_output_directory/"            /*4*/
                  << std::endl;
        return 1;
    }

    std::string output_directory = std::string(argv[4]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[1]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            gaussian_cfg_path, output_dir, 0, device_type);

    // Read the openMVG scene
    pGausMapper->setSensorType(MONOCULAR);
    pGausMapper->setDatasetSourcePath(argv[2]);
    readScene(pGausMapper);
    std::filesystem::path result_ply_path(argv[3]);
    pGausMapper->loadPly(result_ply_path);

    // Test and save results
    pGausMapper->renderAndRecordAllKeyframes("_test");

    return 0;
}