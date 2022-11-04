#include "GPUWeaver.h"

#include <cstdlib>
#include <cmath>
#include <sstream>
#include <chrono>
using namespace std::chrono;

#include "WeaveKernel.h"
#include "lodepng.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

GPUWeaver::GPUWeaver(float* targetImage, Point* points, int resolution, int pointCount, float lineThickness, float gausianBlurRadius)
{
    this->resolution = resolution;
    this->pointCount = pointCount;
    this->currentPoint = 0;
    this->lineThickness = lineThickness;

    connections.push_back(currentPoint);

    initializeGPUMemory();


    // Copy points into cuda memory
    HANDLE_ERROR(cudaMemcpy(d_points, points, pointCount * sizeof(Point), cudaMemcpyHostToDevice));
    // Copy target image into cuda memory
    HANDLE_ERROR(cudaMemcpy(d_targetImage, targetImage, resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));

    // Clear current Image
    float* clearedImage = (float*)malloc(sizeof(float) * resolution * resolution);
    for (size_t y = 0; y < resolution; y++)
    {
        for (size_t x = 0; x < resolution; x++)
        {
            clearedImage[(y * resolution) + x] = 1.0f;
        }
    }
    HANDLE_ERROR(cudaMemcpy(d_currentImage, clearedImage, resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));
    free(clearedImage);

    // Clear current weave block
    float* weaveBlock = (float*)malloc(sizeof(float) * pointCount * resolution * resolution);
    for (size_t z = 0; z < pointCount; z++) 
    {
        for (size_t y = 0; y < resolution; y++)
        {
            for (size_t x = 0; x < resolution; x++)
            {
                weaveBlock[(z * resolution * resolution) + (y * resolution) + x] = 1.0f;
            }
        }
    }
    HANDLE_ERROR(cudaMemcpy(d_weaveBlock, weaveBlock, pointCount * resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_tempWeaveBlock, weaveBlock, pointCount * resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));
    free(weaveBlock);

    // Initalize connection matrix
    h_connectionMatrix = (int*)malloc(pointCount * pointCount * sizeof(int));
    for (size_t x = 0; x < pointCount; x++)
    {
        for (size_t y = 0; y < pointCount; y++)
        {
            // We say that every point is connected to itself to 
            if(x == y)
                h_connectionMatrix[(pointCount * y) + x] = 1;
            else
                h_connectionMatrix[(pointCount * y) + x] = 0;
        }
    }
    HANDLE_ERROR(cudaMemcpy(d_connectionMatrix, h_connectionMatrix, pointCount * pointCount * sizeof(int), cudaMemcpyHostToDevice));

    const float PI = 3.14159265359f;
    int blurRadius = std::max(0, (int)roundf32(resolution * gausianBlurRadius));
    kernelSize = (blurRadius * 2) + 1;
    float* kernel = (float*)malloc(kernelSize * sizeof(float));

    float kernelSum = 0.0f;
    for (int blurX = -blurRadius; blurX <= blurRadius; blurX++)
    {
        float br = blurRadius / 2.5f;
        if(br < 0.001) {
            br = 0.001;
        }
        float strength = (1.0f / (2 * PI * br))
            * expf(- ((blurX * blurX)) / (2.0f * br * br));
        kernel[blurX + blurRadius] = strength;
        kernelSum += strength;
    }
    for (int blurX = -blurRadius; blurX <= blurRadius; blurX++)
    {
        kernel[blurX + blurRadius] /= kernelSum;
        std::cout << kernel[blurX + blurRadius] << std::endl;
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_gausianKernel, kernelSize * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_gausianKernel, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    free(kernel);
}

GPUWeaver::~GPUWeaver()
{
}

void GPUWeaver::makeConnection(int pointIndex) {
    connections.push_back(pointIndex);

    h_connectionMatrix[(pointIndex * resolution) + currentPoint] = 1;
    h_connectionMatrix[(currentPoint * resolution) + pointIndex] = 1;
    HANDLE_ERROR(cudaMemcpy(d_connectionMatrix, h_connectionMatrix, pointCount * pointCount * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_currentImage, &d_weaveBlock[pointIndex * resolution * resolution], (sizeof(float) * resolution * resolution), cudaMemcpyDeviceToDevice));
    currentPoint = pointIndex;
}

float GPUWeaver::weaveIteration() {
    auto begin = high_resolution_clock::now();

    int xLim = resolution;
    int yLim = resolution;
    int zLim = pointCount;  

    const int BLOCK_X = 32;
    const int BLOCK_Y = 32;
    const int BLOCK_Z = 1;

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid((int)ceilf(xLim / BLOCK_X), (int)ceilf(yLim / BLOCK_Y), (int)ceilf(zLim / BLOCK_Z));

    // Start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start);
    // Draw lines
    dev_drawLine<<<grid, block>>>(d_weaveBlock, d_currentImage, d_points, currentPoint, pointCount, resolution, lineThickness);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    // Stop timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); 
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
    printf(" %f ms ", milliseconds);

    // Start timer
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start);
    // Compute loss
    dev_blurX<<<grid, block, BLOCK_X * BLOCK_Y * sizeof(float)>>>(d_weaveBlock, d_tempWeaveBlock, d_gausianKernel, kernelSize, resolution, pointCount);
    dev_calculateLoss<<<grid, block, BLOCK_X * BLOCK_Y * sizeof(float)>>>(d_weaveBlock, d_tempWeaveBlock, d_connectionMatrix, d_currentImage, d_targetImage, d_points, d_scores, d_gausianKernel, kernelSize, currentPoint, pointCount, resolution);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
        // Stop timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); 
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
    printf(" %f ms ", milliseconds);

    float* scores = (float*)malloc(sizeof(float) * pointCount);
    HANDLE_ERROR(cudaMemcpy(scores, d_scores, sizeof(float) * pointCount, cudaMemcpyDeviceToHost));

    // Find min value
    int lowestIndex = 0;
    float minLoss = scores[0];
    for (size_t i = 0; i < pointCount; i++)
    {
        if(scores[i] < minLoss) {
            minLoss = scores[i];
            lowestIndex = i;
        }

        scores[i] = 0;
    }

    // Reset score
    HANDLE_ERROR(cudaMemcpy(d_scores, scores, sizeof(float) * pointCount, cudaMemcpyHostToDevice));
    free(scores);

    makeConnection(lowestIndex);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - begin);
    std::cout << " " << duration.count() / 1000.0f << " ms ";

    return minLoss;
}

void GPUWeaver::saveCurrentImage(const char* fileName) {
    cudaDeviceSynchronize();

    std::cout << "a." << std::endl;
    size_t grayScaleImageSize = sizeof(float) * resolution * resolution;
    float* h_imgData = (float*) malloc(grayScaleImageSize);
    std::cout << "b." << std::endl;
    HANDLE_ERROR(cudaMemcpy(h_imgData, d_currentImage, grayScaleImageSize, cudaMemcpyDeviceToHost));

	size_t outputSize = resolution * resolution * 4 * sizeof(unsigned char);
	std::cout << "e." << std::endl;
    unsigned char* outputData = (unsigned char*)malloc(outputSize);

	for (size_t y = 0; y < resolution; y++)
	{
		for (size_t x = 0; x < resolution; x++)
		{
			int index = ((y * resolution) + x);
			unsigned char val = (unsigned char)(h_imgData[index] * 255);
            val = std::min(std::max((int)val, 0), 0xFF);
			outputData[(index * 4) + 0] = val;
			outputData[(index * 4) + 1] = val;
			outputData[(index * 4) + 2] = val;
			outputData[(index * 4) + 3] = 0xFF;
		}
		
	}

    std::cout << "f." << fileName << " " << (int)outputData[((resolution * resolution * 4) - 1)] << " " << resolution << std::endl;
	lodepng_encode32_file(fileName, outputData, resolution, resolution);
    std::cout << "g." << std::endl;
    // free(h_imgData);
    // free(outputData);
}

std::string GPUWeaver::getInstructionsStr() {
    std::stringstream ss;
    ss.str("");
    ss << connections[0];
    for (auto it = connections.begin() + 1; it != connections.end(); it++)
    {
        ss << " -> " << (*it);
    }
    return ss.str();     
}

void GPUWeaver::initializeGPUMemory() {
    HANDLE_ERROR(cudaMalloc((void**)&d_weaveBlock, resolution * resolution * pointCount * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_tempWeaveBlock, resolution * resolution * pointCount * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_connectionMatrix, pointCount * pointCount * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_currentImage, resolution * resolution * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_points, pointCount * sizeof(Point)));
    HANDLE_ERROR(cudaMalloc((void**)&d_targetImage, resolution * resolution * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_scores, pointCount * sizeof(float)));
}