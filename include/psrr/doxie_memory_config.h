/*!
 * Copyright (c) 2026
 * \file doxie_memory_config.h
 * \brief Shared constants for DOXIE tree memory layout.
 */
#ifndef XGBOOST_TREE_DOXIE_MEMORY_CONFIG_H_
#define XGBOOST_TREE_DOXIE_MEMORY_CONFIG_H_

#include <cstdint>

namespace xgboost {
namespace doxie {

constexpr uint32_t DOXIE_PAGE_SIZE = 4096;
constexpr uint32_t DOXIE_NODE_SIZE = 20;
constexpr uint32_t DOXIE_NODES_PER_PAGE = DOXIE_PAGE_SIZE / DOXIE_NODE_SIZE;
constexpr uint32_t MERGE_LEVEL_LIMIT = 7;

}  // namespace doxie
}  // namespace xgboost

#endif  // XGBOOST_TREE_DOXIE_MEMORY_CONFIG_H_
