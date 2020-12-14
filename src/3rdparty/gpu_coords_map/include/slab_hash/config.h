/*
 * Copyright 2019 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

/** Built-in flags **/
static constexpr uint32_t EMPTY_SLAB_PTR = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_PAIR_PTR = 0xFFFFFFFF;
static constexpr uint32_t HEAD_SLAB_PTR = 0xFFFFFFFE;

/** Queries **/
static constexpr uint32_t SEARCH_NOT_FOUND = 0xFFFFFFFF;

/** Warp operations **/
static constexpr uint32_t WARP_WIDTH = 32;
static constexpr uint32_t BLOCKSIZE_ = 128;

/* bits:   31 | 30 | ... | 3 | 2 | 1 | 0 */
static constexpr uint32_t ACTIVE_LANES_MASK = 0xFFFFFFFF;
static constexpr uint32_t PAIR_PTR_LANES_MASK = 0x7FFFFFFF;
static constexpr uint32_t NEXT_SLAB_PTR_LANE = 31;

using addr_t = uint32_t;

/* These types are all the same, but distiguish the naming can lead to clearer
 * meanings*/
using ptr_t = uint32_t;
static constexpr uint32_t NULL_ITERATOR = 0xFFFFFFFF;
