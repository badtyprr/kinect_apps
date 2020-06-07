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
#include <cstdint>      // Necessary for UINT32_MAX
#include <algorithm>    // provides std::min, std::max
#include <fstream>
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


// Global Variables
VkSurfaceFormatKHR surface_format;
VkPresentModeKHR present_mode;
VkExtent2D extent;
VkSwapchainKHR swap_chain;
std::vector<VkImage> swap_chain_images;
std::vector<VkImageView> swap_chain_image_views;
VkRenderPass render_pass;
VkPipelineLayout pipeline_layout;
VkPipeline graphics_pipeline;


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

static std::vector<char> read_file(const std::string& filename) {
    // ate means seek to the end of the file upon open, and binary just means binary
    // Thanks: https://en.cppreference.com/w/cpp/io/ios_base/openmode
    // The advantage of starting to read at the end of the file is that we can use the read position to determine the size of the file and allocate a buffer.
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::string err_str = "failed to open file!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }

    // tellg returns the position of the current character in the input stream.
    size_t file_size = (size_t)file.tellg();
    // Pre-declare buffer size
    std::vector<char> buffer(file_size);
    // Seek to the beginning
    file.seekg(0);
    // Read all bytes in the file
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
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

    *window = glfwCreateWindow(WIDTH, HEIGHT, "Virtual Green Screen", nullptr, nullptr);
}

void close_window(GLFWwindow* window)
{
    spdlog::get("console")->info("Closing the GLFW window...");
    glfwDestroyWindow(window);
    glfwTerminate();
}

void close_kinect(const k4a_device_t &kinect)
{
    spdlog::get("console")->info("Closing the Kinect camera...");
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
    // TODO: Implement validation callback for non critical errors, see validation layers and swap chain sections for application
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
    spdlog::get("console")->info("Closing the Vulkan instance...");
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

bool has_extensions(const VkPhysicalDevice& physical_device)
{
    spdlog::get("console")->info("Checking for extensions:");
    for (const auto& extension : extensions)
    {
        spdlog::get("console")->info("\t{}", extension);
    }
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());
    /*
     * template< class InputIt >
     * set( InputIt first, InputIt last,
     *    const Compare& comp = Compare(),
     *    const Allocator& alloc = Allocator() );
     */
    std::set<std::string> required_extensions(extensions.begin(), extensions.end());

    for (const auto& extension : available_extensions)
    {
        required_extensions.erase(extension.extensionName);
    }

    if (required_extensions.empty())
    {
        spdlog::get("console")->info("All required extensions were available!");
    }
    return required_extensions.empty();
}

bool has_adequate_basic_swap_chain(const VkPhysicalDevice& physical_device, const VkSurfaceKHR& surface)
{
    SwapChainSupportDetails details = query_swap_chain_support(physical_device, surface);
    return !details.formats.empty() && !details.present_modes.empty() && &details.capabilities != nullptr;
}

bool is_suitable_device(const VkPhysicalDevice &physical_device, const VkSurfaceKHR &surface)
{
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    // Get physical device properties and features
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    vkGetPhysicalDeviceFeatures(physical_device, &features);
    
    // NOTE: It is important that we only try to query for swap chain support after verifying that the extension is available.
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && features.geometryShader && 
        find_queue_families(physical_device, surface).is_complete() && has_extensions(physical_device) && 
        has_adequate_basic_swap_chain(physical_device, surface))
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
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

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
    spdlog::get("console")->info("Closing the Vulkan logical device...");
    vkDestroyDevice(logical_device, nullptr);
}

void close_vulkan_physical_device(const VkPhysicalDevice &physical_device)
{
    spdlog::get("console")->info("Closing the Vulkan physical device...");
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
    spdlog::get("console")->info("Closing the Vulkan surface...");
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

SwapChainSupportDetails query_swap_chain_support(const VkPhysicalDevice& physical_device, const VkSurfaceKHR &surface)
{
    SwapChainSupportDetails details;

    // Query Physical Device Surface Capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &details.capabilities);

    // Query Physical Device Surface Formats
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);

    if (format_count != 0)
    {
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, details.formats.data());
    }

    // Query Physical Device Surface Present Modes
    uint32_t present_modes_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_modes_count, nullptr);

    if (present_modes_count != 0)
    {
        details.present_modes.resize(present_modes_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_modes_count, details.present_modes.data());
    }

    return details;
}

VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
{
    // Choose the color depth and color space
    /*
     * typedef struct VkSurfaceFormatKHR {
     *    VkFormat           format;        // See formats: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkFormat.html
     *    VkColorSpaceKHR    colorSpace;    // See color spaces: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkColorSpaceKHR.html
     * } VkSurfaceFormatKHR;
     */
    if (available_formats.empty())
    {
        std::string err_str = "No available surface formats!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }

    // Try to find a format that's BGRA 32-bit sRGB Nonlinear
    for (const auto& format : available_formats)
    {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return format;
        }
    }

    // Otherwise, return the first one
    return available_formats[0];
}

VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes)
{
    // Choose how images are displayed (i.e. timing)
    /*
     * typedef enum VkPresentModeKHR {
     *  VK_PRESENT_MODE_IMMEDIATE_KHR = 0,
     *  VK_PRESENT_MODE_MAILBOX_KHR = 1,
     *  VK_PRESENT_MODE_FIFO_KHR = 2,
     *  VK_PRESENT_MODE_FIFO_RELAXED_KHR = 3,
     *  VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR = 1000111000,
     *  VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR = 1000111001,
     * } VkPresentModeKHR;
     */

    if (available_present_modes.empty())
    {
        std::string err_str = "No available present modes!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }

    // Search for a better mode
    for (const auto& mode : available_present_modes)
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return mode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    // Choose resolution
    /*
     * typedef struct VkSurfaceCapabilitiesKHR {
     *  uint32_t                         minImageCount;
     *  uint32_t                         maxImageCount;
     *  VkExtent2D                       currentExtent;
     *  VkExtent2D                       minImageExtent;
     *  VkExtent2D                       maxImageExtent;
     *  uint32_t                         maxImageArrayLayers;
     *  VkSurfaceTransformFlagsKHR       supportedTransforms;
     *  VkSurfaceTransformFlagBitsKHR    currentTransform;
     *  VkCompositeAlphaFlagsKHR         supportedCompositeAlpha;
     *  VkImageUsageFlags                supportedUsageFlags;
     * } VkSurfaceCapabilitiesKHR;
     */

    // The maximum is supported
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    else
    {
        VkExtent2D actual_extent = { WIDTH, HEIGHT };
        // Check for extent support
        actual_extent.width = std::max<uint32_t>(capabilities.minImageExtent.width, 
            std::min<uint32_t>(capabilities.maxImageExtent.width, actual_extent.width));
        actual_extent.height = std::max<uint32_t>(capabilities.minImageExtent.height,
            std::min<uint32_t>(capabilities.maxImageExtent.height, actual_extent.height));

        return actual_extent;
    }
}

void initialize_swap_chain(const VkDevice& logical_device, const VkPhysicalDevice& physical_device, const VkSurfaceKHR& surface, VkSwapchainKHR *swap_chain)
{
    SwapChainSupportDetails swap_chain_support = query_swap_chain_support(physical_device, surface);

    // See global variables
    surface_format = choose_swap_surface_format(swap_chain_support.formats);
    present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
    extent = choose_swap_extent(swap_chain_support.capabilities);

    uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;

    if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount)
    {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface;

    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    /*
     * typedef enum VkImageUsageFlagBits {
     *  VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x00000001,
     *  VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002,   // Used for post-processing
     *  VK_IMAGE_USAGE_SAMPLED_BIT = 0x00000004,
     *  VK_IMAGE_USAGE_STORAGE_BIT = 0x00000008,
     *  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x00000010,   // For an image suitable for VkImageView for use in a VkFramebuffer
     *  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020,
     *  VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT = 0x00000040,
     *  VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT = 0x00000080,
     *  VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV = 0x00000100,
     *  VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT = 0x00000200,
     * } VkImageUsageFlagBits;
     */
    // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT specifies that the image can be used to create a VkImageView suitable for use as a color or resolve attachment in a VkFramebuffer.
    // It is also possible that you'll render images to a separate image first to perform operations like post-processing. In that case you may use a value like VK_IMAGE_USAGE_TRANSFER_DST_BIT instead and use a memory operation to transfer the rendered image to a swap chain image.
    
    QueueFamilyIndices indices = find_queue_families(physical_device, surface);
    uint32_t queue_family_indices[] = { indices.graphics_family.value(), indices.presentation_family.value() };

    if (indices.graphics_family != indices.presentation_family) {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queue_family_indices;
    }
    else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0; // Optional
        create_info.pQueueFamilyIndices = nullptr; // Optional
    }

    // NOTE: To specify that you do not want any transformation, simply specify the current transformation.
    create_info.preTransform = swap_chain_support.capabilities.currentTransform;
    
    // NOTE: The compositeAlpha field specifies if the alpha channel should be used for blending with other windows in the window system. You'll almost always want to simply ignore the alpha channel, hence VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR.
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    create_info.presentMode = present_mode;
    // NOTE: If the clipped member is set to VK_TRUE then that means that we don't care about the color of pixels that are obscured.
    create_info.clipped = VK_TRUE;

    // NOTE: With Vulkan it's possible that your swap chain becomes invalid or unoptimized while your application is running, leave as null for now TBD
    create_info.oldSwapchain = VK_NULL_HANDLE;

    // Create the swap chain
    if (vkCreateSwapchainKHR(logical_device, &create_info, nullptr, swap_chain) != VK_SUCCESS)
    {
        std::string err_str = "failed to create swap chain!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }
    else
    {
        spdlog::get("console")->info("Successfully created the swap chain!");
    }
}

void close_swap_chain(const VkDevice& logical_device, const VkSwapchainKHR& swap_chain)
{
    spdlog::get("console")->info("Closing the Vulkan swap chain...");
    vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
}

void get_swap_chain_images(const VkDevice& logical_device, const VkSwapchainKHR& swap_chain, std::vector<VkImage>& swap_chain_images)
{
    uint32_t image_count;
    vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, nullptr);
    swap_chain_images.resize(image_count);
    vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, swap_chain_images.data());
}

void initialize_image_views(const VkDevice& logical_device)
{
    /*
    // Provided by VK_VERSION_1_0
    typedef enum VkImageViewType {
        VK_IMAGE_VIEW_TYPE_1D = 0,
        VK_IMAGE_VIEW_TYPE_2D = 1,
        VK_IMAGE_VIEW_TYPE_3D = 2,
        VK_IMAGE_VIEW_TYPE_CUBE = 3,
        VK_IMAGE_VIEW_TYPE_1D_ARRAY = 4,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY = 5,
        VK_IMAGE_VIEW_TYPE_CUBE_ARRAY = 6,
    } VkImageViewType;
    */
    /* 
    // Provided by VK_VERSION_1_0
    typedef struct VkImageSubresourceRange {
        VkImageAspectFlags    aspectMask;
        uint32_t              baseMipLevel;
        uint32_t              levelCount;
        uint32_t              baseArrayLayer;
        uint32_t              layerCount;
    } VkImageSubresourceRange;
    */
    /*
    // Provided by VK_VERSION_1_0
    typedef enum VkImageAspectFlagBits {
        VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001,
        VK_IMAGE_ASPECT_DEPTH_BIT = 0x00000002,
        VK_IMAGE_ASPECT_STENCIL_BIT = 0x00000004,
        VK_IMAGE_ASPECT_METADATA_BIT = 0x00000008,
        VK_IMAGE_ASPECT_PLANE_0_BIT = 0x00000010,
        VK_IMAGE_ASPECT_PLANE_1_BIT = 0x00000020,
        VK_IMAGE_ASPECT_PLANE_2_BIT = 0x00000040,
        VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT = 0x00000080,
        VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT = 0x00000100,
        VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT = 0x00000200,
        VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT = 0x00000400,
        VK_IMAGE_ASPECT_PLANE_0_BIT_KHR = VK_IMAGE_ASPECT_PLANE_0_BIT,
        VK_IMAGE_ASPECT_PLANE_1_BIT_KHR = VK_IMAGE_ASPECT_PLANE_1_BIT,
        VK_IMAGE_ASPECT_PLANE_2_BIT_KHR = VK_IMAGE_ASPECT_PLANE_2_BIT,
    } VkImageAspectFlagBits;

    */
    swap_chain_image_views.resize(swap_chain_images.size());
    for (size_t i = 0; i < swap_chain_images.size(); i++)
    {
        VkImageViewCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = swap_chain_images[i];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = surface_format.format;
        // Swizzle identity is the same color channel as the image color channel, no change
        create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        // specifying which aspect(s) of the image are included in the view.
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // So, we are only getting RGB values
        create_info.subresourceRange.baseArrayLayer = 0;    // first array layer accessible to the view.
        create_info.subresourceRange.baseMipLevel = 0;  // first mipmap level accessible to the view.
        create_info.subresourceRange.layerCount = 1;    // number of array layers (starting from baseArrayLayer) accessible to the view.
        create_info.subresourceRange.levelCount = 1;    // number of mipmap levels (starting from baseMipLevel) accessible to the view.

        // Create the image view
        if (vkCreateImageView(logical_device, &create_info, nullptr, &swap_chain_image_views[i]) != VK_SUCCESS) 
        {
            std::string err_str = "Failed to create image views!";
            spdlog::get("console")->error(err_str);
            throw std::runtime_error(err_str);
        }
        else
        {
            spdlog::get("console")->info("Successfully create an image view");
        }
    }
}

void close_image_views(const VkDevice& logical_device)
{
    for (const auto& image_view : swap_chain_image_views)
    {
        vkDestroyImageView(logical_device, image_view, nullptr);
        spdlog::get("console")->info("Closed a Vulkan image view");
    }
}

void initialize_graphics_pipeline(const VkDevice& logical_device)
{
    /*
    // Provided by VK_VERSION_1_0
    typedef enum VkShaderStageFlagBits {
        VK_SHADER_STAGE_VERTEX_BIT = 0x00000001,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT = 0x00000002,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT = 0x00000004,
        VK_SHADER_STAGE_GEOMETRY_BIT = 0x00000008,
        VK_SHADER_STAGE_FRAGMENT_BIT = 0x00000010,
        VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
        VK_SHADER_STAGE_ALL_GRAPHICS = 0x0000001F,
        VK_SHADER_STAGE_ALL = 0x7FFFFFFF,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR = 0x00000100,
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR = 0x00000200,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR = 0x00000400,
        VK_SHADER_STAGE_MISS_BIT_KHR = 0x00000800,
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR = 0x00001000,
        VK_SHADER_STAGE_CALLABLE_BIT_KHR = 0x00002000,
        VK_SHADER_STAGE_TASK_BIT_NV = 0x00000040,
        VK_SHADER_STAGE_MESH_BIT_NV = 0x00000080,
        VK_SHADER_STAGE_RAYGEN_BIT_NV = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        VK_SHADER_STAGE_ANY_HIT_BIT_NV = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        VK_SHADER_STAGE_MISS_BIT_NV = VK_SHADER_STAGE_MISS_BIT_KHR,
        VK_SHADER_STAGE_INTERSECTION_BIT_NV = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        VK_SHADER_STAGE_CALLABLE_BIT_NV = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
    } VkShaderStageFlagBits;
    */
    spdlog::get("console")->info("Building the graphics pipeline");

    auto vert_shader_code = read_file("shaders/basic_vert_shader.spv");
    auto frag_shader_code = read_file("shaders/basic_frag_shader.spv");

    spdlog::get("console")->info("Creating the vertex shader");
    VkShaderModule vert_shader_module = create_shader_module(vert_shader_code, logical_device);
    spdlog::get("console")->info("Creating the fragment shader");
    VkShaderModule frag_shader_module = create_shader_module(frag_shader_code, logical_device);

    VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
    vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_info.module = vert_shader_module;
    vert_shader_stage_info.pName = "main";
    // Don't need to specify compile time inputs, set to nullptr
    vert_shader_stage_info.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
    frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_info.module = frag_shader_module;
    frag_shader_stage_info.pName = "main";
    // Don't need to specify compile time inputs, set to nullptr
    frag_shader_stage_info.pSpecializationInfo = nullptr;

    spdlog::get("console")->info("Creating the shader pipeline");
    VkPipelineShaderStageCreateInfo shader_stages[] = { vert_shader_stage_info, frag_shader_stage_info };

    // Vertex Input
    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.pVertexBindingDescriptions = nullptr;
    vertex_input_info.vertexAttributeDescriptionCount = 0;
    vertex_input_info.pVertexAttributeDescriptions = nullptr;

    /*
    // Provided by VK_VERSION_1_0
    typedef struct VkPipelineInputAssemblyStateCreateInfo {
        VkStructureType                            sType;
        const void*                                pNext;
        VkPipelineInputAssemblyStateCreateFlags    flags;
        VkPrimitiveTopology                        topology;
        VkBool32                                   primitiveRestartEnable;
    } VkPipelineInputAssemblyStateCreateInfo;
    */
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    // Viewport
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)extent.width;
    viewport.height = (float)extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    // Scissors
    // While viewports define the transformation from the image to the framebuffer, scissor rectangles define in which regions pixels will actually be stored. 
    // Any pixels outside the scissor rectangles will be discarded by the rasterizer.
    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = extent;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    /*
    // Provided by VK_VERSION_1_0
    typedef struct VkPipelineRasterizationStateCreateInfo {
        VkStructureType                            sType;
        const void*                                pNext;
        VkPipelineRasterizationStateCreateFlags    flags;
        VkBool32                                   depthClampEnable;        // controls whether to clamp the fragment’s depth values as described in Depth Test
        VkBool32                                   rasterizerDiscardEnable; // controls whether primitives are discarded immediately before the rasterization stage.
        VkPolygonMode                              polygonMode;             // is the triangle rendering mode
        VkCullModeFlags                            cullMode;                // triangle facing direction used for primitive culling
        VkFrontFace                                frontFace;               // the front-facing triangle orientation to be used for culling.
        VkBool32                                   depthBiasEnable;         // bias fragment depth values?
        float                                      depthBiasConstantFactor; // scalar factor controlling the constant depth value added to each fragment.
        float                                      depthBiasClamp;          // maximum (or minimum) depth bias of a fragment.
        float                                      depthBiasSlopeFactor;    // scalar factor applied to a fragment’s slope in depth bias calculations
        float                                      lineWidth;               // width of rasterized line segments
    } VkPipelineRasterizationStateCreateInfo;
    */

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;  // geometry never passes through the rasterizer stage, nothing gets rendered
    /*
    VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
    VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
    VK_POLYGON_MODE_POINT: polygon vertices are drawn as points
    */
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f; // fragments, any line thicker than 1.0f requires you to enable the wideLines GPU feature.
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;    // type of face culling to use
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // specifies the vertex order for faces to be considered front-facing and can be clockwise or counterclockwise.
    rasterizer.depthBiasEnable = VK_FALSE;  // can be used for shadow mapping, but not used here
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /*
    // Provided by VK_VERSION_1_0
    typedef struct VkPipelineMultisampleStateCreateInfo {
        VkStructureType                          sType;
        const void*                              pNext;
        VkPipelineMultisampleStateCreateFlags    flags;
        VkSampleCountFlagBits                    rasterizationSamples;
        VkBool32                                 sampleShadingEnable;
        float                                    minSampleShading;
        const VkSampleMask*                      pSampleMask;
        VkBool32                                 alphaToCoverageEnable;
        VkBool32                                 alphaToOneEnable;
    } VkPipelineMultisampleStateCreateInfo;
    */

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    // Color Blending
    // * Mix the old and new value to produce a final color (current)
    // * Combine the old and new value using a bitwise operation (such as alpha blending)
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    // If you want to use the second method of blending (bitwise combination), 
    // then you should set logicOpEnable to VK_TRUE. The bitwise operation can then be specified in the logicOp field. 
    // Note that this will automatically disable the first method, as if you had set blendEnable to VK_FALSE for every attached framebuffer!
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f; // Optional
    color_blending.blendConstants[1] = 0.0f; // Optional
    color_blending.blendConstants[2] = 0.0f; // Optional
    color_blending.blendConstants[3] = 0.0f; // Optional

    // Dynamic State
    VkDynamicState dynamic_states[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_LINE_WIDTH
    };

    // This struct can be substituted by a nullptr later on if you don't have any dynamic state. (I don't)
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    // Uniform values need to be specified during pipeline creation by creating a VkPipelineLayout object
    VkPipelineLayoutCreateInfo pipeline_layout_Info{};
    pipeline_layout_Info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_Info.setLayoutCount = 0; // Optional
    pipeline_layout_Info.pSetLayouts = nullptr; // Optional
    pipeline_layout_Info.pushConstantRangeCount = 0; // Optional
    pipeline_layout_Info.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(logical_device, &pipeline_layout_Info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::string err_str = "failed to create pipeline layout!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }

    // Create Graphics Pipeline
    // All of these combined fully define the functionality of the graphics pipeline, so we can now begin filling in the 
    // VkGraphicsPipelineCreateInfo structure at the end of the createGraphicsPipeline function.But before the calls to 
    // vkDestroyShaderModule because these are still to be used during the creation.
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    // Referencing the array of VkPipelineShaderStageCreateInfo structs
    pipeline_info.stageCount = 2;   // vertex and fragment shaders
    pipeline_info.pStages = shader_stages;
    // fixed-function stage in pipeline_layout
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;    // how elements (like triangles) are constructed
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = nullptr;             // Optional, since we're not using
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = nullptr;                  // Optional, since we're not using, although the structure is setup above
    // Pipeline layout (uniform variables and the like)
    pipeline_info.layout = pipeline_layout;
    // Render pass pipeline, index of the subpass
    // Render pass pipelines must be compatible with subpass pipelines: https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#renderpass-compatibility
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = 0;
    // Vulkan allows you to create a new graphics pipeline by deriving from an existing pipeline. 
    // The idea of pipeline derivatives is that it is less expensive to set up pipelines when they have 
    // much functionality in common with an existing pipeline and switching between pipelines from the same parent can also be done quicker
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;      // Optional
    pipeline_info.basePipelineIndex = -1;                   // Optional

    if (vkCreateGraphicsPipelines(logical_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(logical_device, frag_shader_module, nullptr);
    vkDestroyShaderModule(logical_device, vert_shader_module, nullptr);
}

void close_graphics_pipeline(const VkDevice& logical_device)
{
    spdlog::get("console")->info("Closing the Vulkan graphics pipeline...");
    vkDestroyPipeline(logical_device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(logical_device, pipeline_layout, nullptr);
}

void initialize_render_pass(const VkDevice& logical_device)
{
    VkAttachmentDescription color_attachment{};
    color_attachment.format = surface_format.format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;   // no multisampling
    // The loadOp and storeOp determine what to do with the data in the attachment before rendering and after rendering. 
    /*
    VK_ATTACHMENT_LOAD_OP_LOAD: Preserve the existing contents of the attachment
    VK_ATTACHMENT_LOAD_OP_CLEAR: Clear the values to a constant at the start
    VK_ATTACHMENT_LOAD_OP_DONT_CARE: Existing contents are undefined; we don't care about them
    */
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // clear the framebuffer to black before drawing a new frame
    /*
    VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later
    VK_ATTACHMENT_STORE_OP_DONT_CARE: Contents of the framebuffer will be undefined after the rendering
    */
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // We're interested in seeing the rendered triangle on the screen, so we're going with the store operation here.
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    /*
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: Images to be presented in the swap chain
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: Images to be used as destination for a memory copy operation
    */
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;     // The initialLayout specifies which layout the image will have before the render pass begins
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // The finalLayout specifies the layout to automatically transition to when the render pass finishes

    VkAttachmentReference color_attachment_reference{};
    color_attachment_reference.attachment = 0;
    color_attachment_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;    // graphics, as opposed to compute
    // The index of the attachment in this array is directly referenced from the fragment shader with the layout(location = 0) out vec4 outColor directive!
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_reference;
    /*
    pInputAttachments: Attachments that are read from a shader
    pResolveAttachments: Attachments used for multisampling color attachments
    pDepthStencilAttachment: Attachment for depth and stencil data
    pPreserveAttachments: Attachments that are not used by this subpass, but for which the data must be preserved
    */

    // Render Pass
    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;

    /*
    The VkAttachmentReference does not reference the attachment object directly, it references the index in the attachments array specified in VkRenderPassCreateInfo. 
    This allows you to reuse the same attachments for different purposes.
    Thanks: https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    */
    if (vkCreateRenderPass(logical_device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
        std::string err_str = "failed to create render pass!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }
}

void close_render_pass(const VkDevice& logical_device)
{
    spdlog::get("console")->info("Closing the Vulkan render pass...");
    vkDestroyRenderPass(logical_device, render_pass, nullptr);
}

VkShaderModule create_shader_module(const std::vector<char>& code, const VkDevice& logical_device)
{
    // The bytecode is stored in an std::vector where the default allocator already ensures that the data satisfies the worst case alignment requirements.
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    // the size of the bytecode is specified in bytes, but the bytecode pointer is a uint32_t pointer rather than a char pointer
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shader_module;
    if (vkCreateShaderModule(logical_device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
    {
        std::string err_str = "failed to create shader module!";
        spdlog::get("console")->error(err_str);
        throw std::runtime_error(err_str);
    }

    return shader_module;
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
    //const char** glfwExtensions;
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
    initialize_vulkan_physical_device(instance, &physical_device, surface);
    initialize_vulkan_logical_device(physical_device, &logical_device, &graphics_queue, surface);

    // Initialize the swap chain
    initialize_swap_chain(logical_device, physical_device, surface, &swap_chain);

    // Get swap chain images
    get_swap_chain_images(logical_device, swap_chain, swap_chain_images);
    // Initialize image views
    initialize_image_views(logical_device);
    // Initialize the render pass
    initialize_render_pass(logical_device);
    // Initialize the graphics pipeline
    initialize_graphics_pipeline(logical_device);

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

    spdlog::get("console")->info("Main loop has exited!");

    close_graphics_pipeline(logical_device);
    close_render_pass(logical_device);
    close_image_views(logical_device);
    close_swap_chain(logical_device, swap_chain);
    close_vulkan_logical_device(logical_device);
    close_vulkan_physical_device(physical_device);
    close_surface(instance, surface);
    close_vulkan(instance);
    close_window(window);
    close_kinect(kinect);

    return EXIT_SUCCESS;
}
