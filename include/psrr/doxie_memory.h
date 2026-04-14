/*!
 * Copyright (c) 2026
 * \file doxie_memory.h
 * \brief Memory alignment and mapping for Differentially Oblivious XGBoost Inference.
 */
#ifndef XGBOOST_TREE_DOXIE_MEMORY_H_
#define XGBOOST_TREE_DOXIE_MEMORY_H_

#include <cstdint>
#include <vector>
#include "xgboost/tree_model.h" 

namespace xgboost {
namespace doxie {

// 修改点 1: 加上 DOXIE_ 前缀以避免与操作系统的全局宏产生冲突
constexpr uint32_t DOXIE_PAGE_SIZE = 4096;
constexpr uint32_t DOXIE_NODE_SIZE = 20; 
constexpr uint32_t DOXIE_NODES_PER_PAGE = DOXIE_PAGE_SIZE / DOXIE_NODE_SIZE; // 204 个节点
constexpr uint32_t MERGE_LEVEL_LIMIT = 7; // 前 7 层（层号 0-6）打包在一起

/*!
 * \brief 获取转换所需分配的总 4KB 内存页数
 */
uint32_t CalculateRequiredPages(uint32_t depth);

/*!
 * \brief 将连续的 std::vector 转化为 DOXIE 要求的按层分页对齐内存块
 * 修改点 2: 使用 RegTree::Node 替代裸的 Node
 */
uint8_t* ConvertToAlignedLayout(const std::vector<RegTree::Node>& original_nodes, uint32_t depth);

/*!
 * \brief O(1) 时钟周期的极速物理地址计算函数
 */
inline RegTree::Node* GetNodePtr(uint8_t* base_ptr, uint32_t logical_index) {
    uint32_t d = 31 - __builtin_clz(logical_index + 1);

    // 区域 A：顶层合并区 
    if (d < MERGE_LEVEL_LIMIT) {
        return reinterpret_cast<RegTree::Node*>(base_ptr + logical_index * DOXIE_NODE_SIZE);
    }

    // 区域 B：逐层对齐区
    uint32_t page_idx = 1;
    for (uint32_t k = MERGE_LEVEL_LIMIT; k < d; ++k) {
        page_idx += ((1 << k) + (DOXIE_NODES_PER_PAGE - 1)) / DOXIE_NODES_PER_PAGE; 
    }

    uint32_t idx_in_level = logical_index - ((1 << d) - 1);
    
    uint32_t p_offset = idx_in_level / DOXIE_NODES_PER_PAGE;
    uint32_t n_offset = idx_in_level % DOXIE_NODES_PER_PAGE;

    page_idx += p_offset;

    return reinterpret_cast<RegTree::Node*>(base_ptr + page_idx * DOXIE_PAGE_SIZE + n_offset * DOXIE_NODE_SIZE);
}

} // namespace doxie
} // namespace xgboost

#endif // XGBOOST_TREE_DOXIE_MEMORY_H_