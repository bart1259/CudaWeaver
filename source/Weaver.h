#include <vector>
#include <stdio.h>

#include "GPUUtils.h"
#include "Point.h"
#include "WeaveKernel.h"
#include <iostream>

#ifndef WEAVER_H
#define WEAVER_H

class Weaver
{
public:
    Weaver(float* targetImage, Point* points, int resolution, int pointCount, float lineThickness, float gausianBlurRadius);
    ~Weaver();

    int weaveIteration();
    void makeConnection(int pointIndex);
    void saveCurrentImage(const char* fileName);
    void saveBlockImage(const char* fileName);
private:
    void initializeGPUMemory();

    float* d_weaveBlock;
    float* d_tempWeaveBlock;
    int* d_connectionMatrix;
    float* d_currentImage;
    float* d_targetImage;
    Point* d_points;
    float* d_scores;
    float* d_gausianKernel;

    int* h_connectionMatrix;

    std::vector<int> connections;

    float lineThickness;
    int kernelSize;
    int currentPoint = 0;
    int resolution;
    int pointCount;
};


#endif