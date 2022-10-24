#include "WeaveKernel.h"
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ bool insideLine(float px, float py, float ax, float ay, float bx, float by, float lineThickness) {
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

const int SAMPLE_GRID_SIZE = 8;
// __device__ float d_aa4x[4] = {0.1f, 0.6f, 0.9f, 0.4f};
// __device__ float d_aa4y[4] = {0.6f, 0.9f, 0.4f, 0.41};

__global__ void dev_drawLine(float* d_weaveBlock, 
    float* d_currentImage,
    Point* d_points,
    int* currentPoints,
    int pointCount,
    int resolution,
    float lineThickness,
    Color* colors,
    int colorCount) {
    
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    int colorIndex = z / pointCount;
    int pointIndex = z % pointCount;

    Color color = colors[colorIndex];

    float px = x / (float) resolution;
    float py = y / (float) resolution;

    // Ensure thread is within bound
    if(z < pointCount * colorCount && x < resolution && y < resolution) {

        if(((px - 0.5f) * (px - 0.5f)) + ((py - 0.5f) * (py - 0.5f)) >= 0.25f) {
            return;
        }

        // Draw Line
        float ax = d_points[currentPoints[colorIndex]].x;
        float ay = d_points[currentPoints[colorIndex]].y;
        float bx = d_points[pointIndex].x;
        float by = d_points[pointIndex].y;

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
        
        float alpha = 1 - (val / max);

        d_weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 0] = ((1.0f - alpha) * d_currentImage[(((y * resolution) + x) * 3) + 0]) + (color.r * alpha);
        d_weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 1] = ((1.0f - alpha) * d_currentImage[(((y * resolution) + x) * 3) + 1]) + (color.g * alpha);
        d_weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 2] = ((1.0f - alpha) * d_currentImage[(((y * resolution) + x) * 3) + 2]) + (color.b * alpha);
    }

    return;

}

__device__ int ipowi(int base, int power) {
    int num = 1;
    for (size_t i = 0; i < power; i++)
    {
        num *= base;
    }
    return num;
}

__global__ void dev_calculateLoss(float* d_weaveBlock, 
    float* d_tempWeaveBlock,
    int* d_connectionMatrix,
    float* d_currentImage,
    float* d_targetImage,
    Point* d_points,
    float* d_scores,
    float* d_gausianKernel,
    int kernelSize,
    int* currentPoints,
    int pointCount,
    int resolution,
    Color* colors,
    int colorCount) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    int colorIndex = z / pointCount;
    int pointIndex = z % pointCount;

    // Ensure thread is within bound
    if(z < pointCount * colorCount && x < resolution && y < resolution) {

        // // Blur image
        float accumR = 0.0f;
        float accumG = 0.0f;
        float accumB = 0.0f;
        for (int blurX = -kernelSize / 2; blurX <= kernelSize / 2; blurX++)
        {
            for (int blurY = -kernelSize / 2; blurY <= kernelSize / 2; blurY++)
            {
                float strength = d_gausianKernel[((blurY + kernelSize / 2) * kernelSize) + blurX + (kernelSize / 2)];
                int xi = x + blurX;
                int yi = y + blurY;
                if(xi < 0 || xi >= resolution || yi < 0 || yi >= resolution) {
                    accumR += strength;
                    accumG += strength;
                    accumB += strength;
                } else {
                    accumR += strength * d_weaveBlock[(((z * resolution * resolution) + (yi * resolution) + xi) * 3) + 0];
                    accumG += strength * d_weaveBlock[(((z * resolution * resolution) + (yi * resolution) + xi) * 3) + 1];
                    accumB += strength * d_weaveBlock[(((z * resolution * resolution) + (yi * resolution) + xi) * 3) + 2];
                }
            }
        }
        // float accumR = d_weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 0];
        // float accumG = d_weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 1];
        // float accumB = d_weaveBlock[(((z * resolution * resolution) + (y * resolution) + x) * 3) + 2];


        // Get Pixel loss
        float l1 = fabsf(d_targetImage[(((y * resolution) + x) * 3) + 0] - accumR)
                 + fabsf(d_targetImage[(((y * resolution) + x) * 3) + 1] - accumG)
                 + fabsf(d_targetImage[(((y * resolution) + x) * 3) + 2] - accumB);

        l1 /= 3;
        float l2 = l1 * l1;
        // if(z == 45)
        //     d_currentImage[(y * resolution) + x] = l2;

        float loss = 0.0f;
        loss += l2;

        // If the connection already exists, penalize
        if(d_connectionMatrix[(z * (pointCount * colorCount)) + (currentPoints[colorIndex] + (colorIndex * pointCount))] == 1){
            loss += 1.0f;
        }

        // d_scores[z] = l1 * 10;
        // atomicAdd(&d_scores[z], loss);
        // return;

        // If connection is close, penalize proportionally
        // int apart = 1 + fminf(labs(z - currentPoint), labs(z - 100 - currentPoint));
        // float closenessPenalty = 1.0f / expf(apart);
        // loss += closenessPenalty;

        int idx = (threadIdx.y * blockDim.x) + threadIdx.x;
        __shared__ float blockLoss[32 * 32];
        blockLoss[idx] = loss;

        // Reduce sum
        for (size_t s = 1; s < (blockDim.x * blockDim.y); s *= 2)
        {
            __syncthreads();
            if(idx % (2*s) == 0) {
                blockLoss[idx] += blockLoss[idx + s];
            }
        }
        
        __syncthreads();

        if(threadIdx.x == 0 && threadIdx.y == 0){
            atomicAdd(&d_scores[z], blockLoss[0]);
            // float l = 0.0f;
            // for (size_t a = 0; a < blockDim.x * blockDim.y; a++)
            // {
            //     l+= blockLoss[a];
            // }
            // atomicAdd(&d_scores[z], l);
        }
            // atomicAdd(&d_scores[z], loss);
    }

    // if(z == 49) {
    //     d_currentImage[(y * resolution) + x] = d_tempWeaveBlock[(z * resolution * resolution) + (y * resolution) + x];
    // }
}