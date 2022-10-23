#include "Point.h"
#include "color.h"

#ifndef WEAVE_KERNEL_H
#define WEAVE_KERNEL_H

__global__ void dev_drawLine(float* d_weaveBlock, 
    float* d_currentImage,
    Point* d_points,
    int currentPoint,
    int pointCount,
    int resolution,
    float lineThickness,
    Color* colors,
    int colorCount);

__global__ void dev_calculateLoss(float* d_weaveBlock, 
    float* d_tempWeaveBlock,
    int* d_connectionMatrix,
    float* d_currentImage,
    float* d_targetImage,
    Point* d_points,
    float* d_scores,
    float* d_gausianKernel,
    int kernelSize,
    int currentPoint,
    int pointCount,
    int resolution,
    Color* colors,
    int colorCount);

#endif