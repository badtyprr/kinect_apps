// first_kinect_app.cpp : Opens/closes Kinect, gets RGB/IR/depth frames.
// NuGet dependencies:
// * Microsoft.Azure.Kinect.Sensor
// * spdlog.native

// C++ Libraries
#include <iostream>
// Kinect Libraries
#include <k4a/k4a.h>
// Logging Libraries
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
// User Libraries
#include "errors.hpp"
#include "first_kinect_app.hpp"


void log_frame_info(k4a_image_t image)
{
    int height;
    int width;
    int stride;
    uint64_t timestamp, sys_timestamp;
    uint64_t exposure;
    k4a_image_format_t format;
    uint32_t iso;
    size_t size;
    uint32_t white_balance;

    height = k4a_image_get_height_pixels(image);
    width = k4a_image_get_width_pixels(image);
    stride = k4a_image_get_stride_bytes(image);
    timestamp = k4a_image_get_device_timestamp_usec(image);
    exposure = k4a_image_get_exposure_usec(image);
    format = k4a_image_get_format(image);
    iso = k4a_image_get_iso_speed(image);
    size = k4a_image_get_size(image);
    sys_timestamp = k4a_image_get_system_timestamp_nsec(image);
    white_balance = k4a_image_get_white_balance(image);

    spdlog::get("console")->info("timestamp: {}us, sys timestamp: {}ns, {}x{} frame, format {}, stride {}px, iso {}, exposure {}us, white balance {}K, size {}B",
        timestamp, sys_timestamp, width, height, image_format_to_string[format], stride, iso, exposure, white_balance, size);
}


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
        spdlog::get("console")->error("Attempted to connect with Kinect, and it failed!");
        return KINECT_CONNECT_FAILED;
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
    // 30 fps, 4B per pixel, 1280x720 resolution, WFOV unbinned, synchronized images only (only when all frames are available)
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    config.synchronized_images_only = true;
    // Start Camera
    if (K4A_FAILED(k4a_device_start_cameras(kinect, &config)))
    {
        spdlog::get("console")->error("Attempted to start the cameras, and it failed!");
        return KINECT_START_CAMERAS_FAILED;
    }
    // Capture a depth frame
    unsigned int i = 0;
    k4a_capture_t capture;
    k4a_wait_result_t kinect_wait_result;
    k4a_image_t depth = nullptr, rgb = nullptr, ir = nullptr;
    float temperatureC;
    int width, height, stride;
    uint64_t timestamp, exposure;
    k4a_image_format_t format;
    while (i < N_CAPTURES)
    {
        switch (kinect_wait_result = k4a_device_get_capture(kinect, &capture, CAPTURE_TIMEOUT_MS))
        {
        case K4A_WAIT_RESULT_SUCCEEDED:
            temperatureC = k4a_capture_get_temperature_c(capture);
            spdlog::get("console")->info("Captured frame {}, temperature at {}C", i, temperatureC);
            if ((depth = k4a_capture_get_depth_image(capture)) != nullptr)
            {
                spdlog::get("console")->info("Received Depth frame {}", i);
                log_frame_info(depth);
            }
            if ((rgb = k4a_capture_get_color_image(capture)) != nullptr)
            {
                spdlog::get("console")->info("Received RGB frame {}", i);
                log_frame_info(rgb);
            }
            if ((ir = k4a_capture_get_ir_image(capture)) != nullptr)
            {
                spdlog::get("console")->info("Received IR frame {}", i);
                log_frame_info(ir);
            }

            // Release image object
            k4a_image_release(depth);
            k4a_image_release(rgb);
            k4a_image_release(ir);
            // Release capture object
            k4a_capture_release(capture);
            // Increment frame index
            i++;
            break;
        case K4A_WAIT_RESULT_FAILED:
            spdlog::get("console")->info("FAILED to capture frame {}", i);
            // Capture objects do not need to be released when nothing is captured
            break;
        case K4A_WAIT_RESULT_TIMEOUT:
            spdlog::get("console")->info("TIMED OUT when capturing frame {}", i);
            // Capture objects do not need to be released when nothing is captured
            break;
        default:
            break;
        }
    }

    // Close Cameras
    k4a_device_stop_cameras(kinect);
    // Close Kinect Device
    std::cout << "Closing Kinect device..." << std::endl;
    k4a_device_close(kinect);
}
