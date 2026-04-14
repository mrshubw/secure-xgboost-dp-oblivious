/*!
 * Copyright (c) 2026
 * \file doxie_memory.cc
 */
#include "psrr/doxie_memory.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace xgboost {
namespace doxie {

uint32_t CalculateRequiredPages(uint32_t depth) {
    if (depth == 0) return 0;
    if (depth <= MERGE_LEVEL_LIMIT) return 1; 

    uint32_t total_pages = 1; 
    
    for (uint32_t k = MERGE_LEVEL_LIMIT; k < depth; ++k) {
        uint32_t nodes_in_level = 1 << k; 
        uint32_t pages_for_level = (nodes_in_level + DOXIE_NODES_PER_PAGE - 1) / DOXIE_NODES_PER_PAGE;
        total_pages += pages_for_level;
    }
    
    return total_pages;
}

uint8_t* ConvertToAlignedLayout(const std::vector<RegTree::Node>& original_nodes, uint32_t depth) {
    if (original_nodes.empty()) return nullptr;

    uint32_t total_pages = CalculateRequiredPages(depth);
    size_t total_bytes = total_pages * DOXIE_PAGE_SIZE;

    void* raw_memory = nullptr;
    if (posix_memalign(&raw_memory, DOXIE_PAGE_SIZE, total_bytes) != 0) {
        throw std::runtime_error("DOXIE Error: Failed to allocate page-aligned memory in Enclave.");
    }
    
    uint8_t* aligned_base_ptr = static_cast<uint8_t*>(raw_memory);

    std::memset(aligned_base_ptr, 0, total_bytes);

    uint32_t num_nodes = original_nodes.size(); 
    
    for (uint32_t i = 0; i < num_nodes; ++i) {
        RegTree::Node* dest_ptr = GetNodePtr(aligned_base_ptr, i);
        *dest_ptr = original_nodes[i];
    }

    return aligned_base_ptr;
}

} // namespace doxie
} // namespace xgboost