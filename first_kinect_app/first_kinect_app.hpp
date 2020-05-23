#pragma once

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <map>
#include <string>

//constexpr auto N_CAPTURES = 3;
// Capture timeout minimum is 525 ms.
constexpr auto CAPTURE_TIMEOUT_MS = 578;

constexpr auto RESOLUTION_X = 1280;
constexpr auto RESOLUTION_Y = 720;

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