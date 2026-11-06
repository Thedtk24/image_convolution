#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class ImageProcessor{
    private:
        int width;
        int height;
        int channels;
        vector<unsigned char> h_data;
    
    public:
        ImageProcessor();
        bool load(const string& filepath);
        bool save(const string& filepath) const;
        void toGrayscale();
        void applyCpuConvolution(const vector<float>& kernel, int kernelSize);
        int getWidth() const;
        int getHeight() const;
        int getChannels() const;
        const unsigned char* getData() const;
        vector<unsigned char>& getDataVector();
};

#endif