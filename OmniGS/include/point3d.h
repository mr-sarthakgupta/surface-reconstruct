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

#include <Eigen/Core>

class Point3D
{
public:
    Point3D()
        : xyz_(0.0, 0.0, 0.0),
          color_(0.0f, 0.0f, 0.0f),
          color256_(0, 0, 0),
          error_(-1.0)
    {}

public:
    Eigen::Vector3d xyz_;

    Eigen::Matrix<uint8_t, 3, 1> color256_; // not needed if we get color_ directly
    Eigen::Matrix<float, 3, 1> color_;

    double error_;
};
