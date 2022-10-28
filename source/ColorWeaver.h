#include <vector>
#include <stdio.h>

#include "GPUUtils.h"
#include "Point.h"
#include "color.h"
#include "WeaveKernel.h"
#include "BaseWeaver.h"
#include <iostream>

#ifndef COLOR_WEAVER_H
#define COLOR_WEAVER_H

class ColorWeaver : public BaseWeaver
{
public:
    ColorWeaver(float* targetImage, Point* points, Color* colors, int colorCount, int resolution, int pointCount, float lineThickness, float gausianBlurRadius);
    ~ColorWeaver();

    float weaveIteration();
    void saveCurrentImage(const char* fileName);
    std::string getInstructionsStr();

    void makeConnection(int pointIndex);
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
    Color* d_colors;
    int*   d_currentPoints;

    int* h_connectionMatrix;
    int* h_currentPoints;

    std::vector<int> connections;

    float lineThickness;
    int kernelSize;
    int resolution;
    int pointCount;
    int colorCount;
    int threadsToCheck;
};


#endif