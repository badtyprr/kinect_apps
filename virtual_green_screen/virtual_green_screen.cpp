/* virtual_green_screen.cpp : Masks RGB frames with an aligned depth frame and outputs to a window.
 *
 * NuGet dependencies:
 * - Microsoft.Azure.Kinect.Sensor
 * - spdlog.native
 * - glfw
 * - glm
 *
 * This is probably a helpful reference for Kinect v4 code:
 * - https://github.com/MarekKowalski/LiveScan3D/blob/AzureKinect/src/LiveScanClient/azureKinectCapture.cpp#L205
 * 
 * Reference for point cloud quality:
 * - https://www.youtube.com/watch?v=NrIgjK_PeQU
 * 
 * Libraries for point cloud processing
 * - PCL https://pointclouds.org/
 * - Cilantro https://github.com/kzampog/cilantro (paper: https://arxiv.org/pdf/1807.00399.pdf)
 *
 * "Awesome" Resources:
 * - https://github.com/mmolero/awesome-point-cloud-processing
 */


// Warning suppression
#pragma warning( disable : 26812 )


// C++
#include <iostream>
#include <chrono>
#include <stdexcept>    // provides exception propagation
#include <cstdlib>      // provides EXIT_SUCCESS
#include <optional>     // provides optional data structure
#include <set>          // provides set type
// Vulkan
#include <vulkan/vulkan.h>
// GLFW
#include <GLFW/glfw3.h>
// Kinect
#include <k4a/k4a.h>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// Logging
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
// User
#include "errors.hpp"
#include "virtual_green_screen.hpp"


void log_frame_info(const k4a_image_t &image)
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

    spdlog::get("console")->info("timestamp: {}us, sys timestamp: {}ns, {}x{} frame, format {}, stride {}px, iso {}, exposure {}us, white balance {}K, size {}Bytes",
        timestamp, sys_timestamp, width, height, image_format_to_string[format], stride, iso, exposure, white_balance, size);
}

cv::Mat k4a_to_mat(const k4a_image_t &image)
{
    return cv::Mat(k4a_image_get_height_pixels(image), k4a_image_get_width_pixels(image), CV_8UC4, static_cast<void*>(k4a_image_get_buffer(image)));
}

void display_frame(const cv::Mat &frame)
{
    // Setup Display Window
    cv::namedWindow("RGB Image");
    while (true)
    {
        cv::imshow("RGB Image", frame);
        cv::waitKey(10);
    }
    // Destroy Windows
    cv::destroyAllWindows();
}

void initialize_window(GLFWwindow **window)
{
    if (glfwInit() != GLFW_TRUE)
    {
        spdlog::get("console")->error("GLFW failed to initialize!");
        throw std::runtime_error("GLFW failed to initialize!");
    }

    if (glfwVulkanSupported() == GLFW_TRUE)
        spdlog::get("console")->info("Vulkan is available, at least for compute.");
    else
        spdlog::get("console")->warn("Vulkan is NOT supported!");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_FALSE);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_REFRESH_RATE, 60);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Not OpenGL or OpenGL ES, it's Vulkan!
    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_FALSE);

    *window = glfwCreateWindow(RESOLUTION_X, RESOLUTION_Y, "Virtual Green Screen", nullptr, nullptr);
}

void close_window(GLFWwindow* window)
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void close_kinect(const k4a_device_t &kinect)
{
    // Close Cameras
    k4a_device_stop_cameras(kinect);
    // Close Kinect Device
    std::cout << "Closing Kinect device..." << std::endl;
    k4a_device_close(kinect);
}

void initialize_vulkan(VkInstance *instance)
{
    // Check for validation layer support, if requested
    if (ENABLE_VALIDATION_LAYERS && !check_validation_layer_support())
    {
        std::string err_str = "Validation layers were requested, but one or more layers were not available!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }
    // Application Info
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Virtual Green Screen";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = nullptr;
    appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 0);
    // Patch versions must always be 0, source: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#extendingvulkan-coreversions-versionnumbers
    appInfo.apiVersion = VK_API_VERSION_1_2;   // Macro for VK_MAKE_VERSION(1, 2, 0)

    // Create Info
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (ENABLE_VALIDATION_LAYERS)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    // Use GLFW to get extension count
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    // NOTE: GLFW appears to be returning less extensions (3) vs. Vulkan (13), why?
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;

    // Determines the global validation layers to enable
    createInfo.enabledLayerCount = 0;
    if (vkCreateInstance(&createInfo, nullptr, instance) != VK_SUCCESS)
    {
        std::string err_str = "Failed to create Vulkan instance!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }

    // Determine the number of extensions (according to Vulkan)
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr); // Get all layers and properties
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
    spdlog::get("console")->info("Supports {} Vulkan extensions:", extensionCount);
    for (const auto& extension : extensions)
    {
        spdlog::get("console")->info("\t{}", extension.extensionName);
    }
}

void close_vulkan(const VkInstance &instance)
{
    vkDestroyInstance(instance, nullptr);
}

bool check_validation_layer_support()
{
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    bool layer_found = false;

    for (const auto& layer_name : VALIDATION_LAYERS)
    {
        layer_found = false;
        for (const auto& layer : available_layers)
        {
            if (strcmp(layer_name, layer.layerName) == 0)
            {
                layer_found = true;
                break;
            }
        }

        if (!layer_found)
        {
            spdlog::get("console")->info("Layer {} not found, validation layer check failed!", layer_name);
            return false;
        }
    }

    spdlog::get("console")->info("All validation layers are accounted for!");
    return true;
}

bool is_suitable_device(const VkPhysicalDevice &device, const VkSurfaceKHR &surface)
{
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    // Get physical device properties and features
    vkGetPhysicalDeviceProperties(device, &properties);
    vkGetPhysicalDeviceFeatures(device, &features);
    
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && features.geometryShader && find_queue_families(device, surface).is_complete())
    {

        spdlog::get("console")->info("{} is a suitable device", properties.deviceName);
        return true;
    }
    else
    {
        spdlog::get("console")->info("{} is an unsuitable device", properties.deviceName);
        return false;
    }
}

void initialize_vulkan_physical_device(const VkInstance &instance, VkPhysicalDevice *physical_device, const VkSurfaceKHR &surface)
{
    uint32_t device_count;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
    VkPhysicalDeviceProperties properties;
    for (const auto& device : physical_devices)
    {
        if (is_suitable_device(device, surface))
        {
            *physical_device = device;
            // Get physical device properties
            vkGetPhysicalDeviceProperties(device, &properties);
            spdlog::get("console")->info("Found a suitable device: {}", properties.deviceName);
            break;
        }
    }

    if (*physical_device == VK_NULL_HANDLE)
    {
        std::string err_str = "No suitable GPU device found!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }
}

void initialize_vulkan_logical_device(const VkPhysicalDevice& physical_device, VkDevice* logical_device, VkQueue* queue, const VkSurfaceKHR& surface)
{
    spdlog::get("console")->info("Creating Vulkan logical device...");
    QueueFamilyIndices indices = find_queue_families(physical_device, surface);

    // Create queue info structures
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families;
    try
    {
        indices.start();
        while (true)
        {
            unique_queue_families.insert(indices.next().value());
        }
    }
    catch (std::logic_error e) {}

    float queue_priority = 1.f;

    for (const auto& queue_family : unique_queue_families)
    {
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }


    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount = 0;

    if (ENABLE_VALIDATION_LAYERS)
    {
        create_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }
    else
    {
        create_info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physical_device, &create_info, nullptr, logical_device) != VK_SUCCESS)
    {
        std::string err_str = "Could not create the Vulkan logical device!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }
    else
    {
        spdlog::get("console")->info("Successfully created the logical device!");
    }

    // Gets the graphics queue, and since we only created one queue under the graphics queue, the index is 0
    vkGetDeviceQueue(*logical_device, indices.graphics_family.value(), 0, queue);
}

void close_vulkan_logical_device(const VkDevice &logical_device)
{
    vkDestroyDevice(logical_device, nullptr);
}

void close_vulkan_physical_device(const VkPhysicalDevice &physical_device)
{
    
}

void initialize_surface(const VkInstance &instance, GLFWwindow *window, VkSurfaceKHR *surface)
{
    if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
    {
        std::string err_str = "Failed to create window surface!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }
}

void close_surface(const VkInstance& instance, const VkSurfaceKHR& surface)
{
    vkDestroySurfaceKHR(instance, surface, nullptr);
}

QueueFamilyIndices find_queue_families(const VkPhysicalDevice &device, const VkSurfaceKHR &surface)
{
    QueueFamilyIndices indices;

    // Get Queue Family properties
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    // Find a Queue that supports VK_QUEUE_GRAPHICS_BIT
    int i = 0;
    VkBool32 presentation_supported = false;
    for (const auto& queue_family : queue_families)
    {
        presentation_supported = false;

        // Graphics queue support check
        if (queue_family.queueFlags && VK_QUEUE_GRAPHICS_BIT)
        {
            spdlog::get("console")->info("Found a Graphics capable Queue Family: {}", i);
            indices.graphics_family = i;

            // Presentation queue support check
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentation_supported);

            if (presentation_supported)
            {
                indices.presentation_family = i;
            }
        }

        if (indices.is_complete())
        {
            break;
        }

        i++;
    }

    return indices;
}

int main()
{
    // Setup loggers
    auto stdout_logger = spdlog::stdout_color_mt("console");
    auto stderr_logger = spdlog::stderr_color_mt("stderr");
    spdlog::get("console")->info("Starting Virtual Green Screen Application");

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
    // Stable configurations:
    // 30 fps, Depth WFOV 2x2 Binned, -1000us depth to rgb delay, 578ms timeout
    // 15 fps, Depth WFOV Unbinned, -8000us depth to rgb delay, 630ms timeout
    config.camera_fps = K4A_FRAMES_PER_SECOND_15;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
    config.depth_delay_off_color_usec = -8000;
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
    std::chrono::high_resolution_clock::time_point start_of_frame;
    std::chrono::high_resolution_clock::time_point end_of_frame;
    std::chrono::duration<double, std::milli> duration;

    // Initialize window manager and input control
    spdlog::get("console")->info("Initializing the GLFW window");
    GLFWwindow* window = nullptr;
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    initialize_window(&window);

    // Initialize Vulkan
    spdlog::get("console")->info("Initializing Vulkan");
    VkInstance instance = nullptr;
    initialize_vulkan(&instance);
    // Initialize Surface
    // NOTE: The window surface needs to be created right after the instance creation, because it can actually influence the physical device selection.
    // Thanks: https://vulkan-tutorial.com/en/Drawing_a_triangle/Presentation/Window_surface
    VkSurfaceKHR surface;
    initialize_surface(instance, window, &surface);
    // Initialize physical, logical devices and queues
    VkPhysicalDevice physical_device;
    VkDevice logical_device;
    VkQueue graphics_queue;
    VkQueue presentation_queue;
    initialize_vulkan_physical_device(instance, &physical_device, surface);
    initialize_vulkan_logical_device(physical_device, &logical_device, &graphics_queue, surface);

    while (!glfwWindowShouldClose(window)) 
    {
        // Start of frame timer
        start_of_frame = std::chrono::high_resolution_clock::now();

        glfwPollEvents();

        // Capture Kinect frame bundle (RGB, Depth, IR)
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

            // End of frame timer
            end_of_frame = std::chrono::high_resolution_clock::now();
            // Log the frame period duration
            duration = end_of_frame - start_of_frame;
            spdlog::get("console")->info("Frame {} took {} ms.", i, duration.count());

            // Increment frame index
            i++;
            break;
        case K4A_WAIT_RESULT_FAILED:
            spdlog::get("console")->error("FAILED to capture frame {}", i);
            // Capture objects do not need to be released when nothing is captured
            break;
        case K4A_WAIT_RESULT_TIMEOUT:
            spdlog::get("console")->warn("TIMED OUT when capturing frame {}", i);
            // Capture objects do not need to be released when nothing is captured
            break;
        default:
            break;
        }
    }

    close_vulkan_logical_device(logical_device);
    close_vulkan_physical_device(physical_device);
    close_surface(instance, surface);
    close_vulkan(instance);
    close_window(window);
    close_kinect(kinect);

    return EXIT_SUCCESS;
}
