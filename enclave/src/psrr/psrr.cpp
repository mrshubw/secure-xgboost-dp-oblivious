#include <cassert>
#include "psrr/psrr.h"
#include "enclave/obl_primitives.h"

namespace obl
{
    // 计算标准正态分布的分位数（近似算法）
    double inverseCDF(double p) {
        // 使用近似公式
        if (p < 0.5) {
            return -std::sqrt(-2.0 * std::log(p));
        } else {
            return std::sqrt(-2.0 * std::log(1.0 - p));
        }
    }

    double calculateSigma(double epsilon, double delta, double sensitivity) {
        double c = std::sqrt(2 * std::log(1.25 / delta));
        return c * sensitivity / epsilon;
    }

    double calculateMean(double sigma, double delta) {
        double z = inverseCDF(delta);
        return -sigma * z;
    }

    // Function to generate an n-dimensional array with elements from a standard normal distribution
    inline std::vector<int> generate_normal_distribution_array(int size, double mean, double sigma) {
        std::vector<int> array(size);

        // Seed with a real random value, if available
        std::random_device rd;
        std::mt19937 gen(rd());

        // Standard normal distribution
        std::normal_distribution<> d(mean, sigma);

        for (int i = 0; i < size; ++i) {
            // Sample from the distribution and convert to positive integer
            array[i] = std::abs(static_cast<int>(std::round(d(gen))));
        }

        return array;
    }

    void PSRR::perturb(uint8_t* dummy_examples, size_t item_size, size_t num_dummy_examples){
        // 计算添加的噪声量满足的分布
        sigma = calculateSigma(epsilon, delta, sensitivity);
        mean = calculateMean(sigma, delta);

        // 生成噪声向量
        std::vector<int> noise_vector = generate_normal_distribution_array(num_dummy_examples, mean, sigma);

        // 准备用于存放dummies的空间
        this->item_size = item_size;
        num_dummies = std::accumulate(noise_vector.begin(), noise_vector.end(), 0);
        data.resize(num_dummies * item_size);

        // obliviously 填充dummies
        size_t noise_idx = 0; // 选择数量为noise_vec的第noise_idx个值的dummy example进行填充
        int noise_value = 0; // 选择添加的dummy example的数量
        int count = 0; // 记录当前已经填充索引为dummy_idx的dummy example的数量
        size_t data_idx = 0; // 记录当前已经填充的data的索引
        while (noise_idx < num_dummy_examples)
        {
            // data[data_idx] = dummy_examples[noise_idx], item_size bytes will be copied
            ObliviousArrayAccessBytes(data.data() + data_idx*item_size, dummy_examples, item_size, noise_idx, num_dummy_examples);

            // 用于确定上一步的填充结果是否保留
            // 若已经填充的dummy数量小于需要填充的dummy数量，即count<noise_value，则将上一步的copy结果保留，即data_idx=data_idx+1，否则上一步结果作废，即data_idx=data_idx
            noise_value = ObliviousArrayAccess(noise_vector.data(), noise_idx, noise_vector.size());
            data_idx = ObliviousChoose(count<noise_value, data_idx+1, data_idx);
            count = ObliviousChoose(count<noise_value, count+1, count);

            // 用于确定需要填充的dummy_examples[noise_idx]是否已经完成，下一步是否填充dummy_examples[noise_idx+1]
            noise_idx = ObliviousChoose(count<noise_value, noise_idx, noise_idx+1);
            count = ObliviousChoose(count<noise_value, count, 0);
        }

        // 检查是否已填充所有dummies
        assert(data_idx == num_dummies && "The number of dummies filled does not match the expected amount!");
    }

    void PSRR::shuffle(uint8_t* input, size_t num_items){
        // 复制输入到data
        data.resize((num_items + num_dummies) * item_size);
        std::memcpy(data.data() + num_dummies*item_size, input, num_items * item_size);
        num_real_items = num_items;

        // 调用shuffle函数混洗输入数据与dummy
        shuffler->shuffle(data.data(), num_items + num_dummies, item_size);
    }

    void PSRR::retrieve(std::function<void(uint8_t*, uint8_t*)> func, size_t result_item_size){
        // 对data中的数据循环使用func计算结果
        this->result_item_size = result_item_size;
        result.resize(total_items() * result_item_size);
        for (size_t i = 0; i < total_items(); i++)
        {
            func(data.data() + i*item_size, result.data() + i*result_item_size);
        }
        
    }

    void PSRR::response(uint8_t* output){
        shuffler->inverseShuffle(result.data(), result_item_size, output, num_dummies*result_item_size);
    }
}