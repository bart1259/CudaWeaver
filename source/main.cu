//srun -c 1 -G 1 --pty bash
#include <stdio.h>
#include <memory>
#include <random>
#include <math.h>
#include <vector>
#include <limits>
#include <fstream>
#include <iomanip>

#include "color.h"
#include "Weaver.h"
#include "lodepng.h"

bool loadImage(float** data, uint* width, uint* height, const char* fileName) {
	unsigned char* rawImgData;
	unsigned int error = lodepng_decode32_file(&rawImgData, width, height, fileName);
    if(error) {
        printf("decoder error %u: %s\n", error, lodepng_error_text(error));
        return true;
    }

	*data = (float*)malloc((*width) * (*height) * sizeof(float) * 3);

	for (size_t y = 0; y < *height; y++)
	{
		for (size_t x = 0; x < *width; x++)
		{
			int index = ((y * (*width)) + x);
			(*data)[(index*3) + 0] = rawImgData[(index * 4) + 0] / 255.0f;
			(*data)[(index*3) + 1] = rawImgData[(index * 4) + 1] / 255.0f;
			(*data)[(index*3) + 2] = rawImgData[(index * 4) + 2] / 255.0f;
		}
		
	}

	free(rawImgData);
	return false;
}

float* rescale(float* originalImage, uint width, uint height, uint desiredDim) {
	float* newImage = (float*)malloc(3 * desiredDim * desiredDim * sizeof(float));
	int originalSize = min(width, height);
	float scaling = desiredDim / (float)originalSize;

	for (int y = 0; y < desiredDim; y++)
	{
		for (int x = 0; x < desiredDim; x++)
		{
			int ox = (int)(x / scaling);
			int oy = (int)(y / scaling);
			newImage[(((y * desiredDim) + x) * 3) + 0] = originalImage[(((oy * width) + ox) * 3) + 0];
			newImage[(((y * desiredDim) + x) * 3) + 1] = originalImage[(((oy * width) + ox) * 3) + 1];
			newImage[(((y * desiredDim) + x) * 3) + 2] = originalImage[(((oy * width) + ox) * 3) + 2];
		}
	}

	return newImage;
}

Point* getCircumfrancePoints(int n) {
	const float RADIUS = 0.48f;

	Point* points = (Point*)malloc(sizeof(Point) * n);
	for(int i = 0; i < n; i++) {
		float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float index = i + ((r - 0.5) * 0.4);
		points[i].x = 0.5 + (RADIUS * cosf32( (index / (float)n) * 6.28318530718f));
		points[i].y = 0.5 + (RADIUS * sinf32( (index / (float)n) * 6.28318530718f));
	}

	return points;
}

void printHelp() {
	std::cout << "Weaver usage:" << std::endl
			  << "./weaver [input image] [output image] [instructions output file] [flags]" << std::endl
			  << "  -p The number of points to generate" << std::endl
			  << "  -b The blur radius of the recreated image before the loss calculation (given as a percentage of the total resolution)" << std::endl
			  << "  -i The maximimum number of iterations" << std::endl
			  << "  -l The thickness of the line (given as a percentage of the total resolution)" << std::endl
			  << "  -r The resolution of the image to do the calculations on (in pixels)" << std::endl;
}

int main(int argc, char *argv[]) {
	float* data;
	uint width, height;
	const char* infilename = "input.png";
	const char* outfilename = "output.png";
	const char* instructionsFilePath = nullptr;

	uint pointCount = 102;
	float blurRadius = 0.001f;
	uint iterations = 5000;
	float lineThickness = 0.0005f;
	uint resolution = 512;


	for (size_t i = 1; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
		if(strcmp(argv[i], "-p") == 0) {
			pointCount = atoi(argv[++i]);
			if(pointCount <= 0) {
				printHelp();
				return;
			}
		} else if(strcmp(argv[i], "-b") == 0) {
			blurRadius = (float)atof(argv[++i]);
			if(blurRadius <= 0) {
				printHelp();
				return;
			}
		} else if(strcmp(argv[i], "-i") == 0) {
			iterations = atoi(argv[++i]);
			if(iterations <= 0) {
				printHelp();
				return;
			}
		} else if(strcmp(argv[i], "-l") == 0) {
			lineThickness = (float)atof(argv[++i]);
			if(lineThickness <= 0) {
				printHelp();
				return;
			}
		} else if(strcmp(argv[i], "-r") == 0) {
			resolution = atoi(argv[++i]);
			if(resolution <= 0) {
				printHelp();
				return;
			}
		} else if(i == 1) {
			infilename = argv[i];
		} else if(i == 2) {
			outfilename = argv[i];
		} else if(i == 3) {
			instructionsFilePath = argv[i];
		}
	}
	
	if(loadImage(&data, &width, &height, infilename)) {
		return;
	}

	data = rescale(data, width, height, resolution);

	Point* points = getCircumfrancePoints(pointCount);
	Color colors[] = {
		Color{.r = 1.0f, .g = 0.0f, .b = 0.0f},
		Color{.r = 0.0f, .g = 1.0f, .b = 0.0f},
		Color{.r = 0.0f, .g = 0.0f, .b = 1.0f}
	};
	
	Weaver weaver = Weaver(data, points, colors, 3, resolution, pointCount, lineThickness, blurRadius);
	float prevLoss= std::numeric_limits<float>::max();
	int times = 0;
	const int MAX_FAIL_TIMES = 5;
	for (size_t i = 0; i < iterations; i++)
	{
		float loss = weaver.weaveIteration();
		if (prevLoss - loss < 0.01f) {
			if (++times >= MAX_FAIL_TIMES)
				break;
		} else {
			times = 0;
		}
		std::cout << i << ": " << std::fixed << std::setprecision(3) << loss << std::endl;
		prevLoss = loss;
	}	
	weaver.saveCurrentImage(outfilename);

	if (instructionsFilePath != nullptr) {
		std::ofstream instructionsFile;
		instructionsFile.open(instructionsFilePath);

		instructionsFile << "####### POINTS #######" << std::endl;
		for (size_t i = 0; i < pointCount; i++)
		{
			instructionsFile << i << ", " << points[i].x << ", " << points[i].y << std::endl;
		}
		instructionsFile << "#### INSTRUCTIONS ####" << std::endl;
		instructionsFile <<  weaver.getInstructionsStr() << std::endl;
	}

	return 0;
}