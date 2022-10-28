#include "Point.h"
#include "color.h"
#include "WeaveKernel.h"

#ifndef COLOR_WEAVE_KERNEL_H
#define COLOR_WEAVE_KERNEL_H

__global__ void dev_colorDrawLine(float* d_weaveBlock, 
    float* d_currentImage,
    Point* d_points,
    int* currentPoint,
    int pointCount,
    int resolution,
    float lineThickness,
    Color* colors,
    int colorCount);

__global__ void dev_colorCalculateLoss(float* d_weaveBlock, 
    float* d_tempWeaveBlock,
    int* d_connectionMatrix,
    float* d_currentImage,
    float* d_targetImage,
    Point* d_points,
    float* d_scores,
    float* d_gausianKernel,
    int kernelSize,
    int* currentPoint,
    int pointCount,
    int resolution,
    Color* colors,
    int colorCount);

#endif