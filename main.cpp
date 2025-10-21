#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <execution>
#include "timeit"



std::vector<int> generate_random_data(size_t size) {
    std::vector<int> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 10);
    for (size_t i = 0; i < size; ++i) {
        data[i] = distrib(gen);
    }
    return data;
}

template<typename Iterator, typename OutputIterator, typename T>
void parallel_exclusive_scan(Iterator first, Iterator last, OutputIterator out_first, T init, size_t K) {
    const size_t length = std::distance(first, last);
    if (length == 0) return;

    if (K <= 1 || length < K) {
        std::exclusive_scan(first, last, out_first, init);
        return;
    }

    const size_t block_size = (length + K - 1) / K;
    std::vector<T> block_sums(K);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < K; ++i) {
        threads.emplace_back([=, &block_sums] {
            auto block_start = first;
            std::advance(block_start, i * block_size);
            auto block_end = block_start;
            std::advance(block_end, std::min(block_size, (size_t)std::distance(block_start, last)));

            if (block_start != block_end) {
                block_sums[i] = std::accumulate(block_start, block_end, T{});
            } else {
                block_sums[i] = T{};
            }
        });
    }
    for (auto& t : threads) t.join();
    threads.clear();

    std::vector<T> block_offsets(K);
    std::exclusive_scan(block_sums.begin(), block_sums.end(), block_offsets.begin(), init);

    for (size_t i = 0; i < K; ++i) {
        threads.emplace_back([=, &block_offsets] {
            auto block_start = first;
            std::advance(block_start, i * block_size);
            auto block_end = block_start;
            std::advance(block_end, std::min(block_size, (size_t)std::distance(block_start, last)));

            auto out_block_start = out_first;
            std::advance(out_block_start, i * block_size);

            if (block_start != block_end) {
                std::exclusive_scan(block_start, block_end, out_block_start, block_offsets[i]);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void run_experiments_for_size(size_t data_size) {
    std::cout << "\n=========================================================\n";
    std::cout << "  Test data size = " << data_size << "\n";
    std::cout << "=========================================================\n\n";

    auto data = generate_random_data(data_size);
    std::vector<int> result(data_size);

    std::cout << "std::exclusive_scan (no policy): ";
    timeit([&] {
        std::exclusive_scan(data.begin(), data.end(), result.begin(), 0);
    });




    std::cout << "std::exclusive_scan (seq): ";
    timeit([&] {
        std::exclusive_scan(std::execution::seq, data.begin(), data.end(), result.begin(), 0);
    });



    std::cout << "std::exclusive_scan (par): ";
    timeit([&] {
        std::exclusive_scan(std::execution::par, data.begin(), data.end(), result.begin(), 0);
    });


    std::cout << "std::exclusive_scan (unseq): ";
    timeit([&] {
        std::exclusive_scan(std::execution::unseq, data.begin(), data.end(), result.begin(), 0);
    });



    std::cout << "std::exclusive_scan (par_unseq): ";
    timeit([&] {
        std::exclusive_scan(std::execution::par_unseq, data.begin(), data.end(), result.begin(), 0);
    });



    const size_t hardware_threads = std::thread::hardware_concurrency();
    const size_t max_allowed_threads = hardware_threads != 0 ? hardware_threads : 2;
    for (int i = 2; i <= max_allowed_threads; i++) {
        std::cout << std::format("custom exclusive_scan with {} threads : ", i);
        timeit([&] {
            parallel_exclusive_scan(data.begin(), data.end(), result.begin(), 0, i);
        });

    }

}


int main() {

    std::vector<size_t> data_sizes = {1000000, 10000000, 100000000};

    try {
        for(size_t size : data_sizes) {
            run_experiments_for_size(size);
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << "\n";
        return 1;
    }

    return 0;
}