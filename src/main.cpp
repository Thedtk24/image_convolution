#include <iostream>
#include <vector>
#include "ImageProcessor.hpp"

extern "C" void launchCudaConvolution(const unsigned char* h_in, unsigned char* h_out, int width, int height, const std::vector<float>& kernel, int kernelSize);

int main(int argc, char* argv[]){
    if (argc < 3) {
        std::cerr << "Usage : " << argv[0] << " <image_entree.png> <image_sortie.png>\n";
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    ImageProcessor processor;

    if (!processor.load(inputPath)) {
        return 1;
    }

    processor.toGrayscale();

    int width = processor.getWidth();
    int height = processor.getHeight();

    std::vector<float> kernel = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    };
    int kernelSize = 3;

    const std::vector<unsigned char>& inputData = processor.getDataVector();
    std::vector<unsigned char> outputData(width * height);

    std::cout << "CUDA convolution application..." << std::endl;
    launchCudaConvolution(inputData.data(), outputData.data(), width, height, kernel, kernelSize);

    processor.getDataVector() = std::move(outputData);

    if (!processor.save(outputPath)) {
        return 1;
    }

    std::cout << "Success !" << std::endl;
    return 0;
}