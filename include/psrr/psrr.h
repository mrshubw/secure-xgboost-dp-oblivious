/**
 * @file psrr.h 
 * @brief PSRR is a DP Oblivious algorithm for information retrieval. P: Perturbation, S: Shuffle, R: Retrieval, and R: Response.
 * @author shubowen (shubowen2017@outlook.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once
#include <cstdint>
#include <functional>
#include <vector>
#include "shuffle.h"

namespace obl
{
    class PSRR
    {
    private:
        double epsilon; // privacy budget
        double delta; // privacy budget
        double sensitivity; // sensitivity of the query function
        double sigma; // noise level added to the query function
        double mean; // mean of noise added to the query function

        std::vector<uint8_t> data; // input data
        size_t item_size; // size of each item
        size_t num_dummies = 0; // number of dummies to add, which are randomly drawn from the dummy examples
        size_t num_real_items = 0; // number of real items
        std::vector<uint8_t> result; // output data of retrieval
        size_t result_item_size; // size of each item in the output data

        std::unique_ptr<OShuffler> shuffler; // shuffler used in shuffle() and response()
    public:
        PSRR(double epsilon, double delta, double sensitivity, std::string shuffle_method="BitonicShuffler") : epsilon(epsilon), delta(delta), sensitivity(sensitivity) {
            shuffler = create(shuffle_method);
        };
        size_t total_items() const { return num_real_items + num_dummies; }

        void perturb(uint8_t* dummy_examples, size_t item_size, size_t num_dummy_examples);
        void shuffle(uint8_t* input, size_t num_items);
        void retrieve(std::function<void(uint8_t*, uint8_t*)> func, size_t result_item_size);
        void response(uint8_t* output);
    };
    
}
