#version 450
#extension GL_ARB_separate_shader_objects : enable

// location is the index of the framebuffer
// outColor is specified, and not a magic name
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    // Puts out the color red
    outColor = vec4(fragColor, 1.0);
}