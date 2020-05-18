// first_kinect_app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// C++ Libraries
#include <iostream>
// Kinect Libraries
#include <k4a/k4a.h>
// Logging Libraries
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
// User Libraries
#include "errors.h"

int main()
{
    // Setup loggers
    auto stdout_logger = spdlog::stdout_color_mt("console");
    auto stderr_logger = spdlog::stderr_color_mt("stderr");
    spdlog::get("console")->info("Starting First Kinect Application");
    // Determines how many Kinect devices are installed
    uint32_t count = k4a_device_get_installed_count();
    spdlog::get("console")->info("There are {} Kinect devices on your computer.", count);
    k4a_device_t kinect = nullptr;
    // Connect to the Kinect device that comes up first
    // Results: K4A_RESULT_SUCCEEDED = 0, K4A_RESULT_FAILED = Non-zero
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &kinect))) {
        spdlog::get("console")->info("Attempted to connect with Kinect, and it failed!");
        return 0;
    }
    else
        spdlog::get("console")->info("Attempted to connect with Kinect, and it succeeded!");
    
    // Get the size of the serial number
    size_t serial_size = 0;
    k4a_device_get_serialnum(kinect, NULL, &serial_size);
    // Allocate a char buffer
    char* serial_number = new char[serial_size];
    k4a_device_get_serialnum(kinect, serial_number, &serial_size);
    spdlog::get("console")->info("Opened Kinect Serial Number {}", serial_number);
    delete[] serial_number;

    // Configure the Camera
    // Initialize with no image format, RGB resolution, depth depth mode, 2 fps, no wired sync, and a few other parameters that aren't obvious
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    // 30 fps, 4B per pixel, 1280x720 resolution
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    // Start Camera
    k4a_device_start_cameras(kinect, &config);



    // Close Cameras
    k4a_device_stop_cameras(kinect);
    // Close Kinect Device
    std::cout << "Closing Kinect device..." << std::endl;
    k4a_device_close(kinect);


}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
