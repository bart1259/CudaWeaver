#include "CPUWeaver.h"

#include <cstdlib>
#include <cmath>
#include <sstream>
#include <chrono>
using namespace std::chrono;

#include "lodepng.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

CPUWeaver::CPUWeaver(float* targetImage, Point* points, int resolution, int pointCount, float lineThickness, float gausianBlurRadius)
{
    this->resolution = resolution;
    this->pointCount = pointCount;
    this->currentPoint = 0;
    this->lineThickness = lineThickness;

    connections.push_back(currentPoint);


    this->h_points = points;
    this->h_targetImage = targetImage;

    // Clear current Image
    this->h_currentImage = (float*)malloc(sizeof(float) * resolution * resolution);
    for (size_t y = 0; y < resolution; y++)
    {
        for (size_t x = 0; x < resolution; x++)
        {
            this->h_currentImage[(y * resolution) + x] = 1.0f;
        }
    }

    // Clear current weave block
    this->h_weaveBlock = (float*)malloc(sizeof(float) * pointCount * resolution * resolution);
    for (size_t z = 0; z < pointCount; z++) 
    {
        for (size_t y = 0; y < resolution; y++)
        {
            for (size_t x = 0; x < resolution; x++)
            {
                this->h_weaveBlock[(z * resolution * resolution) + (y * resolution) + x] = 1.0f;
            }
        }
    }

    // Initalize connection matrix
    this->h_connectionMatrix = (int*)malloc(pointCount * pointCount * sizeof(int));
    for (size_t x = 0; x < pointCount; x++)
    {
        for (size_t y = 0; y < pointCount; y++)
        {
            // We say that every point is connected to itself to 
            if(x == y)
                this->h_connectionMatrix[(pointCount * y) + x] = 1;
            else
                this->h_connectionMatrix[(pointCount * y) + x] = 0;
        }
    }

    const float PI = 3.14159265359f;
    int blurRadius = roundf32(resolution * gausianBlurRadius);
    this->kernelSize = (blurRadius * 2) + 1;
    this->h_gausianKernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));

    float kernelSum = 0.0f;
    for (int blurX = -blurRadius; blurX <= blurRadius; blurX++)
    {
        for (int blurY = -blurRadius; blurY <= blurRadius; blurY++)
        {
            float strength = (1.0f / (2 * PI * blurRadius * blurRadius))
                * expf(- ((blurX * blurX) + (blurY * blurY)) / (2.0f * blurRadius * blurRadius));
            this->h_gausianKernel[((blurY + blurRadius) * kernelSize) + blurX + blurRadius] = strength;
            kernelSum += strength;
        }
    }
    for (int blurX = -blurRadius; blurX <= blurRadius; blurX++)
    {
        for (int blurY = -blurRadius; blurY <= blurRadius; blurY++)
        {
            this->h_gausianKernel[((blurY + blurRadius) * kernelSize) + blurX + blurRadius] /= kernelSum;
        }
    }
}

CPUWeaver::~CPUWeaver()
{
    free(this->h_currentImage);
    free(this->h_connectionMatrix);
    free(this->h_gausianKernel);
}

const int SAMPLE_GRID_SIZE = 8;

void CPUWeaver::makeConnection(int pointIndex) {
    connections.push_back(pointIndex);

    h_connectionMatrix[(pointIndex * pointCount) + currentPoint] = 1;
    h_connectionMatrix[(currentPoint * pointCount) + pointIndex] = 1;
    memcpy(this->h_currentImage, &this->h_weaveBlock[pointIndex * resolution * resolution], sizeof(float) * resolution * resolution);
    currentPoint = pointIndex;
}

int CPUWeaver::weaveIteration() {
    auto begin = high_resolution_clock::now();
    timespec ts, te; 
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts); 

    dev_drawLine();

    clock_gettime(CLOCK_MONOTONIC_RAW, &te); 
    printf(" %f ms ", cpu_time(&ts, &te));
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts); 

    dev_calculateLoss();

    clock_gettime(CLOCK_MONOTONIC_RAW, &te); 
    printf(" %f ms ", cpu_time(&ts, &te));

    // Find min value
    int lowestIndex = 0;
    float minLoss = h_scores[0];
    for (size_t i = 0; i < pointCount; i++)
    {
        if(h_scores[i] < minLoss) {
            minLoss = h_scores[i];
            lowestIndex = i;
        }

        h_scores[i] = 0;
    }

    makeConnection(lowestIndex);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - begin);
    std::cout << " " << duration.count() / 1000.0f << " ms ";

    return minLoss;
}

void CPUWeaver::saveCurrentImage(const char* fileName) {

    std::cout << "a." << std::endl;
    std::cout << "b." << std::endl;

	size_t outputSize = resolution * resolution * 4 * sizeof(unsigned char);
	std::cout << "e." << std::endl;
    unsigned char* outputData = (unsigned char*)malloc(outputSize);

	for (size_t y = 0; y < resolution; y++)
	{
		for (size_t x = 0; x < resolution; x++)
		{
			int index = ((y * resolution) + x);
			unsigned char val = (unsigned char)std::min(std::max((int)(h_currentImage[index] * 255), 0), 0xFF);
			outputData[(index * 4) + 0] = val;
			outputData[(index * 4) + 1] = val;
			outputData[(index * 4) + 2] = val;
			outputData[(index * 4) + 3] = 0xFF;
		}
		
	}

    std::cout << "f." << fileName << " " << (int)outputData[((resolution * resolution * 4) - 1)] << " " << resolution << std::endl;
	lodepng_encode32_file(fileName, outputData, resolution, resolution);
    std::cout << "g." << std::endl;
    free(outputData);
}

std::string CPUWeaver::getInstructionsStr() {
    std::stringstream ss;
    ss.str("");
    ss << connections[0];
    for (auto it = connections.begin() + 1; it != connections.end(); it++)
    {
        ss << " -> " << (*it);
    }
    return ss.str();     
}

float CPUWeaver::cpu_time(timespec* start, timespec* end){ 
    return ((1e9*end->tv_sec + end->tv_nsec) - (1e9*start->tv_sec + start->tv_nsec))/1e6; 
} 

void CPUWeaver::dev_drawLine(){
    for (int z = 0; z < pointCount; z++){
        for (int x = 0; x < resolution; x++){
            for (int y = 0; y < resolution; y++){
                float px = x / (float) resolution;
                float py = y / (float) resolution;

                if(((px - 0.5f) * (px - 0.5f)) + ((py - 0.5f) * (py - 0.5f)) < 0.25f) {
                    // Draw Line
                    float ax = h_points[currentPoint].x;
                    float ay = h_points[currentPoint].y;
                    float bx = h_points[z].x;
                    float by = h_points[z].y;

                    float val = 0.0f;
                    float max = 0.0f;

                    // Perform antialiasing
                    for (size_t xa = 0; xa < SAMPLE_GRID_SIZE; xa++)
                    {
                        for (size_t ya = 0; ya < SAMPLE_GRID_SIZE; ya++)
                        {
                            if(((ya * SAMPLE_GRID_SIZE) + xa) % 5 == 0) {
                                float pxaa = px + (xa / (float)SAMPLE_GRID_SIZE / resolution);
                                float pyaa = py + (ya / (float)SAMPLE_GRID_SIZE / resolution);
                                if(!insideLine(pxaa, pyaa, ax, ay, bx, by, lineThickness)) {
                                    val += 1.0f;
                                }
                                max += 1.0f;
                            }
                        }
                    }
                    
                    val /= max;
                    h_weaveBlock[(z * resolution * resolution) + (y * resolution) + x] = h_currentImage[(y * resolution) + x] * val;
                }
            }
        }
    }
} 

void CPUWeaver::dev_calculateLoss(){
    for (int z = 0; z < pointCount; z++){
        for (int x = 0; x < resolution; x++){
            for (int y = 0; y < resolution; y++){
                // // Blur image
                float accum = 0.0f;
                for (int blurX = -kernelSize / 2; blurX <= kernelSize / 2; blurX++)
                {
                    for (int blurY = -kernelSize / 2; blurY <= kernelSize / 2; blurY++)
                    {
                        float strength = h_gausianKernel[((blurY + kernelSize / 2) * kernelSize) + blurX + (kernelSize / 2)];
                        int xi = x + blurX;
                        int yi = y + blurY;
                        if(xi < 0 || xi >= resolution || yi < 0 || yi >= resolution) {
                            accum += strength;
                        } else {
                            accum += strength * h_weaveBlock[(z * resolution * resolution) + (yi * resolution) + xi];
                        }
                    }
                }

                // Get Pixel loss
                float l1 = accum - h_targetImage[(y * resolution) + x];
                float l2 = l1 * l1;

                float loss = 0.0f;
                loss += l2;

                // If the connection already exists, penalize
                if(h_connectionMatrix[(z * resolution) + currentPoint] == 1){
                    loss += 1.0f;
                }

                // If connection is close, penalize proportionally
                int apart = 1 + fminf(labs(z - currentPoint), labs(z - 100 - currentPoint));
                float closenessPenalty = 1.0f / expf(apart);
                loss += closenessPenalty;

                h_scores[z] += loss;
            }
        }
    }
}

bool CPUWeaver::insideLine(float px, float py, float ax, float ay, float bx, float by, float lineThickness) {
    float vAPx = px - ax;
    float vAPy = py - ay;
    float vABx = bx - ax;
    float vABy = by - ay;

    float sqDist = (vABx * vABx) + (vABy * vABy);
    float abaProd = (vABx * vAPx) + (vABy * vAPy);
    float amount = abaProd / sqDist;

    amount = fminf(fmaxf(amount, 0.0f), 1.0f);

    float nx = (amount * (bx - ax)) + ax;
    float ny = (amount * (by - ay)) + ay;

    float dist = sqrtf(((py - ny) * (py - ny)) + ((px - nx) * (px - nx)));
    return dist < lineThickness / 2.0f;
}
