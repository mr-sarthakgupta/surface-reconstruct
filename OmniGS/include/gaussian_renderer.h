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

#include "sh_utils.h"

#include "gaussian_parameters.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_rasterizer.h"

class GaussianRenderer
{
public:
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        bool render_depth = false,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> renderLonlat(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);
};
