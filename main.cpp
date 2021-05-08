#include "jetson_clocks/jetson_clocks.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
    uid_t euid = geteuid();
    if (euid != 0) {
        std::cout << "Must run as root." << std::endl;
        return 1;
    }

    std::vector<long int> freqs = jetson_clocks::get_gpu_available_freqs();
    for(long int f : freqs) {
        printf("%li\n", f);
    }

            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 1], freqs[freqs.size() - 1]);
            return 0;

    for(int i = 0; i < 60; i++) {
        int gpu_load = jetson_clocks::get_gpu_current_usage() / 10;

        // Need to make the actual chart, numbers are there as placeholders for now.
        if (gpu_load > 95) {
            // Max
            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 1], freqs[freqs.size() - 1]);
        } else if (gpu_load > 90) {
            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 2], freqs[freqs.size() - 2]);
        } else if (gpu_load > 85) {
            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 3], freqs[freqs.size() - 3]);
        } else if (gpu_load > 80) {
            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 4], freqs[freqs.size() - 4]);
        } else if (gpu_load > 75) {
            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 5], freqs[freqs.size() - 5]);
        } else {
            // Set to the optimal frequency
            jetson_clocks::set_gpu_freq_range(freqs[freqs.size() - 6], freqs[freqs.size() - 6]);
        }

        // TODO
        // int cpu_load = jetson_clocks::get_cpu_current_usage();

        printf("Load: %d%%\n", gpu_load);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}