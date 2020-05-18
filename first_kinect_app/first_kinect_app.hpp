#pragma once

#include <map>
#include <string>

#define N_CAPTURES 30
// Capture timeout minimum is 525 ms.
#define CAPTURE_TIMEOUT_MS 578

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