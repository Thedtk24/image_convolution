# CUDA / C++ 2D Image Convolution Filter

An image processing application implementing a **2D convolution filter**.

## Prerequisites & Dependencies

* **C++ Compiler** with C++17 support
* **CMake** (version 3.18 or higher)
* **Third-Party Libraries**:
  * [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h) and [stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h) by Sean Barrett (nothings/stb). Included directly in `include/`.

## Build & Execution

```bash
module purge
module load gcc
module load cuda
module load cmake

mkdir -p build && cd build
cmake ..
make
```

To execute on a CUDA GPU compute node, request an interactive Slurm session:

```bash
srun --partition=<gpu_partition> --gres=gpu:1 --pty --ntasks=1 --cpus-per-task=4 --time=00:30:00 bash
./image_filter input.png output_image.png
```

## Credits & Acknowledgments

* **STB Single-File Public Domain Libraries**: Image loading (`stb_image.h`) and writing (`stb_image_write.h`) are authored by **Sean Barrett** ([@nothings](https://github.com/nothings)) and distributed under the MIT / Public Domain licenses via the [nothings/stb repository](https://github.com/nothings/stb).