#pragma once

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <map>
#include <string>

/*
 * Constants
 */

//constexpr auto N_CAPTURES = 3;
// Capture timeout minimum is 525 ms.
constexpr auto CAPTURE_TIMEOUT_MS = 630;

constexpr auto RESOLUTION_X = 1280;
constexpr auto RESOLUTION_Y = 720;

/*
 * Data
 */

std::map<k4a_image_format_t, std::string> image_format_to_string = {
	{K4A_IMAGE_FORMAT_COLOR_MJPG, "MJPG"},
	{K4A_IMAGE_FORMAT_COLOR_NV12, "NV12"},
	{K4A_IMAGE_FORMAT_COLOR_YUY2, "YUY2"},
	{K4A_IMAGE_FORMAT_COLOR_BGRA32, "BGRA32"},
	{K4A_IMAGE_FORMAT_DEPTH16, "DEPTH16"},
	{K4A_IMAGE_FORMAT_IR16, "IR16"},
	{K4A_IMAGE_FORMAT_CUSTOM8, "CUSTOM8"},
	{K4A_IMAGE_FORMAT_CUSTOM16, "CUSTOM16"},
	{K4A_IMAGE_FORMAT_CUSTOM, "CUSTOM"}
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> presentation_family;
    uint i = 0;

    /*
     * @return whether or not a queue family is a complete chain
     */
    bool is_complete() { return graphics_family.has_value() && presentation_family.has_value(); }

    /*
     * Iterator for the next element
     * @return a queue family
     */
    std::optional<uint32_t> next()
    {
        switch (i)
        {
        case 0:
            i++;
            return graphics_family;
        case 1:
            i++;
            return presentation_family;
        default:
            throw std::logic_error("Reached the end of the iterator");
        }
    }

    /* 
     * Resets the iterator to the start
     */
    void start()
    {
        i = 0;
    }
};

/*
 * Functions
 */

void log_frame_info(const k4a_image_t &image);
cv::Mat k4a_to_mat(const k4a_image_t &image);
void display_frame(const cv::Mat& frame);
void initialize_window(GLFWwindow **window);
void close_window(GLFWwindow *window);
void close_kinect(const k4a_device_t& kinect);
bool check_validation_layer_support();
void initialize_vulkan(VkInstance* instance);
void initialize_vulkan_physical_device(const VkInstance& instance, VkPhysicalDevice* physical_device, const VkSurfaceKHR& surface);
void initialize_vulkan_logical_device(const VkPhysicalDevice& physical_device, VkDevice* logical_device, VkQueue* queue, const VkSurfaceKHR& surface);
void close_vulkan_logical_device(const VkDevice& logical_device);
void close_vulkan_physical_device(const VkPhysicalDevice& physical_device);
void initialize_surface(const VkInstance& instance, GLFWwindow *window, VkSurfaceKHR* surface);
bool is_suitable_device(const VkPhysicalDevice& device, const VkSurfaceKHR& surface);

/*
 * Finds the first queue family with a Graphics queue.
 * @param device the physical device associated with the current Vulkan instance
 * @return a QueueFamilyIndices struct that optionally contains the first queue family index with a Graphics queue
 */
QueueFamilyIndices find_queue_families(const VkPhysicalDevice& device, const VkSurfaceKHR& surface);

/*
 * Debug
 */
const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif