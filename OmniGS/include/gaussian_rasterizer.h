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

#pragma once

#include <tuple>

#include <torch/torch.h>

#include "camera.h"
#include "rasterize_points.h"
#include "gaussian_model.h"

struct GaussianRasterizationSettings
{
    GaussianRasterizationSettings(
        int image_height,
        int image_width,
        float tanfovx,
        float tanfovy,
        torch::Tensor& bg,
        float scale_modifier,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix,
        torch::Tensor& RwcT,
        int sh_degree,
        torch::Tensor& campos,
        bool prefiltered,
        Camera::CameraModelType camera_type = Camera::CameraModelType::PINHOLE,
        bool render_depth = false)
        : image_height_(image_height), image_width_(image_width), tanfovx_(tanfovx), tanfovy_(tanfovy),
          bg_(bg), scale_modifier_(scale_modifier), viewmatrix_(viewmatrix), projmatrix_(projmatrix), RwcT_(RwcT),
          sh_degree_(sh_degree), campos_(campos), prefiltered_(prefiltered), camera_type_(camera_type),
          render_depth_(render_depth)
    {}

    int image_height_;
    int image_width_;
    float tanfovx_;
    float tanfovy_;
    torch::Tensor bg_;
    float scale_modifier_;
    torch::Tensor viewmatrix_;
    torch::Tensor projmatrix_;
    torch::Tensor RwcT_;
    int sh_degree_;
    torch::Tensor campos_;
    bool prefiltered_;
    Camera::CameraModelType camera_type_;
    bool render_depth_;
};

class GaussianRasterizerFunction : public torch::autograd::Function<GaussianRasterizerFunction>
{
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3Ds_precomp,
        GaussianRasterizationSettings raster_settings);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_out_color);
};

inline torch::autograd::tensor_list rasterizeGaussians(
    torch::Tensor& means3D,
    torch::Tensor& means2D,
    torch::Tensor& sh,
    torch::Tensor& colors_precomp,
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& cov3Ds_precomp,
    GaussianRasterizationSettings& raster_settings)
{
    return GaussianRasterizerFunction::apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings
    );
}

class GaussianRasterizer : public torch::nn::Module
{
public:
    GaussianRasterizer(GaussianRasterizationSettings& raster_settings)
        : raster_settings_(raster_settings)
    {}

    torch::Tensor markVisibleGaussians(torch::Tensor& positions);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        bool has_shs,
        bool has_colors_precomp,
        bool has_scales,
        bool has_rotations,
        bool has_cov3D_precomp,
        torch::Tensor shs,
        torch::Tensor colors_precomp,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3D_precomp);

public:
    GaussianRasterizationSettings raster_settings_;
};
