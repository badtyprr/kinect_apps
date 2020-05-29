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

    bool is_complete() { return graphics_family.has_value(); }
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
void initialize_vulkan_device(const VkInstance& instance, VkPhysicalDevice* physical_device);
void close_vulkan_device();
bool is_suitable_device(const VkPhysicalDevice& device);
QueueFamilyIndices find_queue_families(const VkPhysicalDevice& device);

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