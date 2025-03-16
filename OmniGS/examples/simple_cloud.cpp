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

constexpr int IMAGE_WIDTH = 2000;
constexpr int IMAGE_HEIGHT = 1000;


void readViewsOpenMVG(
    std::shared_ptr<GaussianMapper> pMapper)
{
    // Cameras
    class Camera camera;
    camera.camera_id_ = 0;

    camera.setModelId(Camera::CameraModelType::LONLAT);
    camera.width_ = IMAGE_WIDTH;
    camera.height_ = IMAGE_HEIGHT;

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

    // Frames
    std::size_t fid = 0;

    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(fid);
    pkf->img_filename_ = "simple_cloud.png";
    pkf->zfar_ = pMapper->z_far_;
    pkf->znear_ = pMapper->z_near_;

    // Pose
    Eigen::Vector3f ccw = Eigen::Vector3f::Zero();
    Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tcw = -Rcw * ccw;
    pkf->setPose(
        Eigen::Quaterniond(Rcw.cast<double>()),
        tcw.cast<double>());

    try {
        // Camera
        unsigned int camera_id = 0;
        Camera& camera = pMapper->scene_->cameras_.at(camera_id);
        std::size_t width = IMAGE_WIDTH;
        std::size_t height = IMAGE_HEIGHT;
        assert(width == camera.width_ && height == camera.height_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
        cv::Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
        cv::cvtColor(image, image, CV_BGR2RGB);
        image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
        cv::Mat imgRGB_undistorted;
        camera.undistortImage(image, imgRGB_undistorted);
        pkf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image, pMapper->device_type_);
        pMapper->scene_->addKeyframe(pkf, &pMapper->kfid_shuffled_);
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::keyframesFromJson]KeyFrame Camera not found!");
    }
}

void readPoints3DOpenMVG(
    std::shared_ptr<GaussianScene> scene, float dist)
{
    class Point3D point3D;

    point3D_id_t point3D_id;
    point3D_id = 0;
    point3D.xyz_(0) = dist;
    point3D.xyz_(1) = -5*dist;
    point3D.xyz_(2) = dist;
    point3D.color256_(0) = 255;
    point3D.color256_(1) = 0;
    point3D.color256_(2) = 0;
    point3D.color_(0) = point3D.color256_(0) / 255.0f;
    point3D.color_(1) = point3D.color256_(1) / 255.0f;
    point3D.color_(2) = point3D.color256_(2) / 255.0f;
    scene->cachePoint3D(point3D_id, point3D);

    point3D_id = 1;
    point3D.xyz_(0) = -dist;
    point3D.xyz_(1) = 0.5*dist;
    point3D.xyz_(2) = -0.7*dist;
    point3D.color256_(0) = 0;
    point3D.color256_(1) = 255;
    point3D.color256_(2) = 0;
    point3D.color_(0) = point3D.color256_(0) / 255.0f;
    point3D.color_(1) = point3D.color256_(1) / 255.0f;
    point3D.color_(2) = point3D.color256_(2) / 255.0f;
    scene->cachePoint3D(point3D_id, point3D);

    point3D_id = 2;
    point3D.xyz_(0) = dist;
    point3D.xyz_(1) = dist;
    point3D.xyz_(2) = -dist;
    point3D.color256_(0) = 0;
    point3D.color256_(1) = 0;
    point3D.color256_(2) = 255;
    point3D.color_(0) = point3D.color256_(0) / 255.0f;
    point3D.color_(1) = point3D.color256_(1) / 255.0f;
    point3D.color_(2) = point3D.color256_(2) / 255.0f;
    scene->cachePoint3D(point3D_id, point3D);

}

void readScene(std::shared_ptr<GaussianMapper> pMapper, float dist)
{
    auto& model_params = pMapper->getGaussianModelParams();
    auto scene = pMapper->scene_;

    readViewsOpenMVG(pMapper);
    readPoints3DOpenMVG(scene, dist);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_output_directory/"            /*2*/
                  << " dist"                                 /*3*/
                  << std::endl;
        return 1;
    }

    std::string output_directory = std::string(argv[2]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    float dist = std::stof(std::string(argv[3]));

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

    // Read the colmap scene
    pGausMapper->setSensorType(MONOCULAR);
    readScene(pGausMapper, dist);

    // Create and save results
    pGausMapper->gaussians_->createFromPcd(
        pGausMapper->scene_->cached_point_cloud_, 1.0f);
    pGausMapper->gaussians_->scaling_ = -0.3*torch::ones_like(pGausMapper->gaussians_->scaling_);
    pGausMapper->gaussians_->opacity_ = 5*torch::ones_like(pGausMapper->gaussians_->opacity_);
    pGausMapper->renderAndRecordAllKeyframes();

    return 0;
}