#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include "../include/ImageProcessor.hpp"

using namespace std;

ImageProcessor::ImageProcessor() : width(0), height(0), channels(0) {}

bool ImageProcessor::load(const string& filepath){
    unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    if(!img){
        cerr << "Error : failure to load image " << filepath << endl;
        return false;
    }

    h_data.assign(img, img + (width * height * channels));
    stbi_image_free(img);
    cout << "Image loaded successfully : " << width << "x" << height << " (" << channels << " channels)" << endl;
    return true;
}

bool ImageProcessor::save(const string& filepath) const{
    int success = stbi_write_png(filepath.c_str(), width, height, channels, h_data.data(), width * channels);
    if(!success){
        cerr << "Error : failure to save image " << filepath << endl;
        return false;
    }
    cout << "Image saved : " << filepath << endl;
    return true;
}

void ImageProcessor::toGrayscale(){
    if (channels == 1) return; 

    std::vector<unsigned char> grayData(width * height);
    for(int i = 0; i < width * height; ++i){
        unsigned char r = h_data[i * channels + 0];
        unsigned char g = h_data[i * channels + 1];
        unsigned char b = h_data[i * channels + 2];
        grayData[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }

    h_data = std::move(grayData);
    channels = 1;
}

void ImageProcessor::applyCpuConvolution(const std::vector<float>& kernel, int kernelSize){
    if(channels != 1){
        cerr << "Error : filter need grey image." << endl;
        return;
    }

    vector<unsigned char> output(width * height, 0);
    int half = kernelSize / 2;

    for(int y = 0; y < height; ++y){
        for(int x = 0; x < width; ++x){
            float sum = 0.0f;
            for(int ky = -half; ky <= half; ++ky){
                for(int kx = -half; kx <= half; ++kx){
                    int px = min(max(x + kx, 0), width - 1);
                    int py = min(max(y + ky, 0), height - 1);
                    
                    float pixelValue = static_cast<float>(h_data[py * width + px]);
                    float weight = kernel[(ky + half) * kernelSize + (kx + half)];
                    sum += pixelValue * weight;
                }
            }
            output[y * width + x] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }
    h_data = move(output);
}

int ImageProcessor::getWidth() const{
    return width;
}

int ImageProcessor::getHeight() const{
    return height;
}
        
int ImageProcessor::getChannels() const{
    return channels;
}

const unsigned char* ImageProcessor::getData() const{
    return h_data.data();
}

vector<unsigned char>& ImageProcessor::getDataVector(){
    return h_data;
}