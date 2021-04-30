#include "jetson_clocks/jetson_clocks.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

class Stats {
    private:
        int gpu_usage;
        bool should_run;
    public:
        Stats() {
            this->should_run = true;
        }
        void setGpuUsage(int usage) { this->gpu_usage = usage; }
        int getGpuUsage() { return this->gpu_usage; }
        bool shouldRun() { return this->should_run; }
        void stopRunning() { this->should_run = false; }
};

void listener(Stats* keeper) {

    while(keeper->shouldRun()) {
        std::cout << "Checking" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

int main(int argc, char* argv[]) {
    uid_t euid = geteuid();
    if (euid != 0) {
        std::cout << "Must run as root." << std::endl;
        return 1;
    }

    Stats* stat_keeper = new Stats();
    std::thread t1(listener, stat_keeper);

    return 0;
}