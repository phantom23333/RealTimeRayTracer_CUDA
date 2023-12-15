# RealTimeRayTracer_CUDA

## Overview
RealTimeRayTracer_CUDA is a real-time ray tracing application developed using CUDA. It draws inspiration from [Ray Tracing in One Weekend in CUDA](https://github.com/rogerallen/raytracinginoneweekendincuda) and showcases the capabilities of CUDA in rendering complex scenes with various materials and lighting models in real-time.

### Features
- **Real-Time Rendering**: Leveraging CUDA for high-performance graphics computations.
- **Camera Movement**: Interactive camera controls for exploring scenes.
- **Material Models**: Including Microfacet, Glossy, Dielectric, and Diffuse.
- **Frame Blending**: Utilizes a mix of the current and last frames for smoother rendering.
- **Challenging Implementations**: Tackles complex topics like BVH (Bounding Volume Hierarchy), although currently not fully implemented.

## Prerequisites
- CUDA Toolkit 12.3
- GLM (OpenGL Mathematics)
- GLFW (Graphics Library Framework)

## Installation
To set up the RealTimeRayTracer_CUDA, follow these steps:

1. Clone the repository
2. using Cmake to build the project

## Demo
Check the [YouTube demo]([youtube-link-here](https://www.youtube.com/video/49SIkR3sGUk)) to see RealTimeRayTracer_CUDA in action.

## Challenges and Limitations
- **BVH Implementation**: Implementing the BVH node in CUDA is complex, and it's currently a work-in-progress.
- **Emissive Objects**: All objects in the world are considered as emitting light, which may not be physically accurate for all scenarios.
- **Frame Blending Technique**: The application uses a blend of the current and the previous frame to achieve smoother rendering transitions.

## Features

Support four shading model
- **Glossy**
- **Diffuse**
- **Dieletric**
- **Microfacet**

Denoise 
- **Temporal** : a simple blend of current frame and last frame was implemented in order to denoise

Loading Mesh
- **Load_Triangle** : using Load_triangle.hpp to load custom .obj meshes. However, it's need to init on host constructor and then copy the memory to the device memory. 

