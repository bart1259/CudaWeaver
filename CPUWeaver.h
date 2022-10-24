#include <vector>
#include <stdio.h>

#include "GPUUtils.h"
#include "Point.h"
#include <iostream>

#ifndef CPUWEAVER_H
#define CPUWEAVER_H

class CPUWeaver
{
public:
    CPUWeaver(float* targetImage, Point* points, int resolution, int pointCount, float lineThickness, float gausianBlurRadius);
    ~CPUWeaver();

    int weaveIteration();
    void makeConnection(int pointIndex);
    void saveCurrentImage(const char* fileName);
    void saveBlockImage(const char* fileName);
    float cpu_time(timespec* start, timespec* end);
    void dev_drawLine();
    void dev_calculateLoss();
    bool insideLine(float px, float py, float ax, float ay, float bx, float by, float lineThickness);
    std::string getInstructionsStr();
private:
    float* h_weaveBlock;
    float* h_tempWeaveBlock;
    int* h_connectionMatrix;
    float* h_currentImage;
    float* h_targetImage;
    Point* h_points;
    float* h_scores;
    float* h_gausianKernel;

    std::vector<int> connections;

    float lineThickness;
    int kernelSize;
    int currentPoint = 0;
    int resolution;
    int pointCount;
};


#endif
