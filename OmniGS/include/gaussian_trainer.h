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

#include <torch/torch.h>

#include <iomanip>
#include <random>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "loss_utils.h"
#include "gaussian_parameters.h"
#include "gaussian_model.h"
#include "gaussian_scene.h"
#include "gaussian_renderer.h"


class GaussianTrainer
{
public:
    GaussianTrainer();

    static void trainingOnce(
        std::shared_ptr<GaussianScene> scene,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianModelParams& dataset,
        GaussianOptimizationParams& opt,
        GaussianPipelineParams& pipe,
        torch::DeviceType device_type = torch::kCUDA,
        std::vector<int> testing_iterations = {},
        std::vector<int> saving_iterations = {},
        std::vector<int> checkpoint_iterations = {}/*, checkpoint*/);

    static void trainingReport(
        int iteration,
        int num_iterations,
        torch::Tensor& Ll1,
        torch::Tensor& loss,
        float ema_loss_for_log,
        std::function<torch::Tensor(torch::Tensor&, torch::Tensor&)> l1_loss,
        int64_t elapsed_time,
        GaussianModel& gaussians,
        GaussianScene& scene,
        GaussianPipelineParams& pipe,
        torch::Tensor& background);

};
