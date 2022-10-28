#include "ColorWeaver.h"

#include <cstdlib>
#include <cmath>
#include <sstream>
#include <chrono>
using namespace std::chrono;

#include "ColorWeaverKernel.h"
#include "lodepng.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ColorWeaver::ColorWeaver(float* targetImage, Point* points, Color* colors, int colorCount, int resolution, int pointCount, float lineThickness, float gausianBlurRadius) : BaseWeaver()
{
    this->resolution = resolution;
    this->pointCount = pointCount;
    this->lineThickness = lineThickness;
    this->colorCount = colorCount;
    this->threadsToCheck = colorCount * pointCount;

    initializeGPUMemory();

    // Copy colors into cuda memory
    HANDLE_ERROR(cudaMemcpy(d_colors, colors, colorCount * sizeof(Color), cudaMemcpyHostToDevice));
    // Copy points into cuda memory
    HANDLE_ERROR(cudaMemcpy(d_points, points, pointCount * sizeof(Point), cudaMemcpyHostToDevice));
    // Copy target image into cuda memory
    HANDLE_ERROR(cudaMemcpy(d_targetImage, targetImage, 3 * resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));

    // Send current points to GPU
    h_currentPoints = (int*)malloc(sizeof(int) * colorCount);
    for (size_t i = 0; i < colorCount; i++)
    {
        // Initialize where each thread begins
        h_currentPoints[i] = i * pointCount / colorCount;
    }
    HANDLE_ERROR(cudaMemcpy(d_currentPoints, h_currentPoints, sizeof(int) * colorCount, cudaMemcpyHostToDevice));

    // Clear current Image
    float* clearedImage = (float*)malloc(sizeof(float) * 3 * resolution * resolution);
    for (size_t y = 0; y < resolution; y++)
    {
        for (size_t x = 0; x < resolution; x++)
        {
            for (size_t t = 0; t < 3; t++)
            {
                clearedImage[(((y * resolution) + x) * 3) + 0] = 1.0f;
                clearedImage[(((y * resolution) + x) * 3) + 1] = 1.0f;
                clearedImage[(((y * resolution) + x) * 3) + 2] = 1.0f;
            }
        }
    }
    HANDLE_ERROR(cudaMemcpy(d_currentImage, clearedImage, 3 * resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));
    free(clearedImage);

    // Clear current weave block
    float* weaveBlock = (float*)malloc(pointCount * colorCount * resolution * resolution * 3 * sizeof(float));
    for (size_t z = 0; z < pointCount * colorCount; z++) 
    {
        for (size_t y = 0; y < resolution; y++)
        {
            for (size_t x = 0; x < resolution; x++)
            {
                weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 0] = 1.0f;
                weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 1] = 1.0f;
                weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 2] = 1.0f;
            }
        }
    }
    HANDLE_ERROR(cudaMemcpy(d_weaveBlock,     weaveBlock, pointCount * colorCount * resolution * resolution * 3 * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_tempWeaveBlock, weaveBlock, pointCount * colorCount * resolution * resolution * 3 * sizeof(float), cudaMemcpyHostToDevice));
    free(weaveBlock);

    // Initalize connection matrix
    h_connectionMatrix = (int*)malloc((pointCount * colorCount) * (pointCount * colorCount) * sizeof(int));
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
    HANDLE_ERROR(cudaMemcpy(d_connectionMatrix, h_connectionMatrix, (pointCount * colorCount) * (pointCount * colorCount) * sizeof(int), cudaMemcpyHostToDevice));

    const float PI = 3.14159265359f;
    int blurRadius = roundf32(resolution * gausianBlurRadius);
    kernelSize = (blurRadius * 2) + 1;
    float* kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));

    float kernelSum = 0.0f;
    for (int blurX = -blurRadius; blurX <= blurRadius; blurX++)
    {
        for (int blurY = -blurRadius; blurY <= blurRadius; blurY++)
        {
            float strength = (1.0f / (2 * PI * blurRadius * blurRadius))
                * expf(- ((blurX * blurX) + (blurY * blurY)) / (2.0f * blurRadius * blurRadius));
            kernel[((blurY + blurRadius) * kernelSize) + blurX + blurRadius] = strength;
            kernelSum += strength;
        }
    }
    for (int blurX = -blurRadius; blurX <= blurRadius; blurX++)
    {
        for (int blurY = -blurRadius; blurY <= blurRadius; blurY++)
        {
            kernel[((blurY + blurRadius) * kernelSize) + blurX + blurRadius] /= kernelSum;
        }
    }
    HANDLE_ERROR(cudaMalloc((void**)&d_gausianKernel, kernelSize * kernelSize * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_gausianKernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    free(kernel);
}

ColorWeaver::~ColorWeaver()
{
}

void ColorWeaver::makeConnection(int pointIndex) {
    connections.push_back(pointIndex);
    int colorIndex = pointIndex / pointCount;
    int colorPoint = h_currentPoints[colorIndex];

    h_connectionMatrix[(pointIndex * (colorCount * pointCount)) + ((colorIndex * pointCount) + colorPoint)] = 1;
    h_connectionMatrix[(((colorIndex * pointCount) + colorPoint) * (colorCount * pointCount)) + pointIndex] = 1;
    HANDLE_ERROR(cudaMemcpy(d_connectionMatrix, h_connectionMatrix, (pointCount * colorCount) * (pointCount * colorCount) * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_currentImage, &d_weaveBlock[pointIndex * resolution * resolution * 3], (sizeof(float) * 3 * resolution * resolution), cudaMemcpyDeviceToDevice));
    h_currentPoints[colorIndex] = pointIndex % pointCount;
    HANDLE_ERROR(cudaMemcpy(d_currentPoints, h_currentPoints, sizeof(int) * colorCount, cudaMemcpyHostToDevice));
}

float ColorWeaver::weaveIteration() {
    auto begin = high_resolution_clock::now();

    int xLim = resolution;
    int yLim = resolution;
    int zLim = pointCount * colorCount;  

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
    dev_colorDrawLine<<<grid, block>>>(d_weaveBlock, d_currentImage, d_points, d_currentPoints, pointCount, resolution, lineThickness, d_colors, colorCount);
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
    dev_colorCalculateLoss<<<grid, block, BLOCK_X * BLOCK_Y * sizeof(float)>>>(d_weaveBlock, d_tempWeaveBlock, d_connectionMatrix, d_currentImage, d_targetImage, d_points, d_scores, d_gausianKernel, kernelSize, d_currentPoints, pointCount, resolution, d_colors, colorCount);
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
    HANDLE_ERROR(cudaMemcpy(scores, d_scores, sizeof(float) * pointCount * colorCount, cudaMemcpyDeviceToHost));

    // Find min value
    int lowestIndex = 0;
    float minLoss = scores[0];
    for (size_t i = 0; i < pointCount * colorCount; i++)
    {
        if(scores[i] < minLoss) {
            minLoss = scores[i];
            lowestIndex = i;
        }

        scores[i] = 0;
    }

    // Reset score
    HANDLE_ERROR(cudaMemcpy(d_scores, scores, sizeof(float) * pointCount * colorCount, cudaMemcpyHostToDevice));
    free(scores);

    makeConnection(lowestIndex);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - begin);
    std::cout << " " << duration.count() / 1000.0f << " ms ";

    return minLoss;
}

void ColorWeaver::saveCurrentImage(const char* fileName) {
    cudaDeviceSynchronize();

    std::cout << "a." << std::endl;
    size_t imageSize = sizeof(float) * resolution * resolution * 3;
    float* h_imgData = (float*) malloc(imageSize);
    std::cout << "b." << std::endl;
    HANDLE_ERROR(cudaMemcpy(h_imgData, d_currentImage, imageSize, cudaMemcpyDeviceToHost));

	size_t outputSize = resolution * resolution * 4 * sizeof(unsigned char);
	std::cout << "e." << std::endl;
    unsigned char* outputData = (unsigned char*)malloc(outputSize);

	for (size_t y = 0; y < resolution; y++)
	{
		for (size_t x = 0; x < resolution; x++)
		{
			int index = ((y * resolution) + x);
			outputData[(index * 4) + 0] = min(max((unsigned char)(h_imgData[(index * 3) + 0] * 255),0),0xFF);
			outputData[(index * 4) + 1] = min(max((unsigned char)(h_imgData[(index * 3) + 1] * 255),0),0xFF);
			outputData[(index * 4) + 2] = min(max((unsigned char)(h_imgData[(index * 3) + 2] * 255),0),0xFF);
			outputData[(index * 4) + 3] = 0xFF;
		}
		
	}

    std::cout << "f." << fileName << " " << (int)outputData[((resolution * resolution * 4) - 1)] << " " << resolution << std::endl;
	lodepng_encode32_file(fileName, outputData, resolution, resolution);
    std::cout << "g." << std::endl;
    free(h_imgData);
    free(outputData);
}

std::string ColorWeaver::getInstructionsStr() {
    std::stringstream ss;
    ss.str("");
    ss << connections[0];
    for (auto it = connections.begin() + 1; it != connections.end(); it++)
    {
        ss << " -> " << (*it);
    }
    return ss.str();     
}

void ColorWeaver::initializeGPUMemory() {
    HANDLE_ERROR(cudaMalloc((void**)&d_weaveBlock, 3 * resolution * resolution * pointCount * colorCount * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_tempWeaveBlock, 3 * resolution * resolution * pointCount * colorCount * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_connectionMatrix, (pointCount * colorCount) * (pointCount * colorCount) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_currentImage, 3 * resolution * resolution * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_points, pointCount * sizeof(Point)));
    HANDLE_ERROR(cudaMalloc((void**)&d_targetImage, 3 * resolution * resolution * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_scores, colorCount * pointCount * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_colors, colorCount * sizeof(Color)));
    HANDLE_ERROR(cudaMalloc((void**)&d_currentPoints, colorCount * sizeof(int)));
}