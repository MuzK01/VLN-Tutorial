# MatterPort3D Simulator Tutorial

This tutorial guides you through setting up and using the MatterPort3D Simulator for Vision-and-Language Navigation (VLN) tasks.
This tutorial is tailored from [Matterport3DSimulatory](https://github.com/peteanderson80/Matterport3DSimulator) with serval modifications.

## Prerequisites
- Docker ([Installation Guide](https://docs.docker.com/get-docker/))
- NVIDIA Docker ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Git

## 1. Repository Setup

### 1.1 Clone the Repository
```bash
git clone https://github.com/MuzK01/VLN-Tutorial.git
cd VLN-Tutorial
```

### 1.2 Download MP3D Dataset
```bash
# Download a single test scene
python scripts/download_mp3d.py -o data --id mJXqzFtmKg4

# Download the complete dataset (1.3TB)
python scripts/download_mp3d.py -o data
```

### 1.3 Download R2R Dataset
```bash
bash scripts/download.sh #it will download the R2R dataset to the R2R_benchmark/data/ directory
```

## 2. Environment Setup

You can set up the environment in two ways: using Docker (recommended) or building from source.

### 2.1 Docker Setup (Recommended)

#### 2.1.1 Build or Pull Docker Image
```bash
# Option 1: Build from Dockerfile
docker build -t mattersim:v1 .

# Option 2: Pull pre-built image file
# [Link to be added]
```

Verify the image is available:
```bash
docker images | grep mattersim
```

#### 2.1.2 Run Docker Container
```bash
docker run -it --gpus all \
    --privileged \
    --shm-size=32g \
    --network=host \
    --device=/dev/video* \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /absolute/path/to/VLN-Tutorial:/projects/VLN-Tutorial \
    mattersim:v1
```

#set default python
```
ln -sf /usr/bin/python3 /usr/bin/python
```

**Docker Arguments Explained:**
- `--gpus all`: Enable GPU support
- `--privileged`: Grant extended privileges
- `--shm-size=32g`: Set shared memory size
- `--network=host`: Use host network
- `--device=/dev/video*`: Mount video devices
- `-e DISPLAY=$DISPLAY`: Enable GUI applications
- `-v /tmp/.X11-unix:/tmp/.X11-unix`: Mount X11 socket
- `-v /absolute/path/to/VLN-Tutorial:/VLN-Tutorial`: Mount project directory

### 2.2 Building from Source

#### 2.2.1 System Requirements
- Ubuntu ≥ 14.04
- NVIDIA drivers with CUDA
- C++11 compatible compiler
- CMake ≥ 3.10
- OpenCV ≥ 2.4
- OpenGL
- GLM
- NumPy

#### 2.2.2 Install Dependencies
```bash
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
```

## 3. Building MatterSim Package
Note that the code of Matterport3DSimulator is modifed to fit to new linux verions: specifcally, docker image version ubuntu 20.04+cuda 11.2. It is different from the original code. But it is possible to still work for other versions.
```bash
ln -sf /usr/bin/python3 /usr/bin/python
cd /projects/VLN-Tutorial/Matterport3DSimulator
mkdir build && cd build
which python #check python is available
cmake -DEGL_RENDERING=ON ..
make
cd ../../
```

### 3.1 Rendering Options
- Default GPU rendering (OpenGL): `cmake ..`
- Off-screen GPU rendering (EGL): `cmake -DEGL_RENDERING=ON ..`
- Off-screen CPU rendering (OSMesa): `cmake -DOSMESA_RENDERING=ON ..`

### 3.2 Configure PYTHONPATH
```bash
# Temporary setup
export PYTHONPATH=/projects/VLN-Tutorial/Matterport3DSimulator/build:$PYTHONPATH

# Permanent setup (add to ~/.bashrc)
echo "export PYTHONPATH=/projects/VLN-Tutorial/Matterport3DSimulator/build:\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

### 3.3 Set EGL_PLATFORM
```bash
echo "export EGL_PLATFORM=device" >> ~/.bashrc
source ~/.bashrc
```

### 3.3 Verify Installation
```bash
python -c "import MatterSim; print('Import MatterSim successfully')"
```

## 4. Using MatterSim

### 4.1 Run Eval
```bash
python seq2seq/eval.py
```

### 4.2 Run training

1. Download the preprocessed image features 
In our initial work using this simulator, we discretized heading and elevation into 30 degree increments, and precomputed image features for each view. Now that the simulator is much faster, this is no longer necessary, but for completeness we include the details of this setting below.

We generate image features using Caffe. To replicate our approach, first download and save some Caffe ResNet-152 weights into the `models` directory. We experiment with weights pretrained on [ImageNet](https://github.com/KaimingHe/deep-residual-networks), and also weights finetuned on the [Places365](https://github.com/CSAILVision/places365) dataset. The script `scripts/precompute_features.py` can then be used to precompute ResNet-152 features. Features are saved in tsv format in the `img_features` directory. 

Alternatively, skip the generation and just download and extract our tsv files into the `img_features` directory:
- [ResNet-152-imagenet features [380K/2.9GB]](https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1)
- [ResNet-152-places365 features [380K/2.9GB]](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1)

2. Run the training script
```bash
python seq2seq/train.py
```

### 4.2 Homework: Check these functions for more details
1. How the Simulator is initialized: Located in `R2R_benchmark/env.py` (`R2RBatch.__init__`)
2. How Scene Navigation Graphs are define: Located in `Matterport3DSimulator/connectivity/` and `R2R_benchmark/utils.py load_nav_graphs`
3. How the episode data is loaded: Located in `R2R_benchmark/utils.py load_datasets`
4. Observation Space: Defined in `R2R_benchmark/env.py` (`R2RBatch._get_obs`)
3. How Teaching Actions are generated: Implemented in `R2R_benchmark/env.py` (`R2RBatch._shortest_path_action`)

## Additional Resources
- [MatterPort3D Dataset](https://niessner.github.io/Matterport/)
- [Original Simulator Repository](https://github.com/peteanderson80/Matterport3DSimulator)

## 5. Troubleshooting

### 5.1 ImportError: No module named MatterSim
If you encounter this error, check the following:

1. **Python Version Mismatch**
   - Ensure the Python version used for building matches the runtime Python version
   - Verify with:
   ```bash
   python --version  # Should match the version used during build
   ```
   - Rebuild if necessary using:
   ```bash
   cmake -DPYTHON_EXECUTABLE=$(which python) -DEGL_RENDERING=ON ..
   make
   ```

2. **PYTHONPATH Configuration**
   - Verify the build directory is in PYTHONPATH:
   ```bash
   echo $PYTHONPATH  # Should include /VLN-Tutorial/Matterport3DSimulator/build
   ```
   - If missing, add it:
   ```bash
   export PYTHONPATH=/VLN-Tutorial/Matterport3DSimulator/build:$PYTHONPATH
   ```

3. **Check Build Output**
   - Verify the .so file exists:
   ```bash
   ls /VLN-Tutorial/Matterport3DSimulator/build/MatterSim*.so
   ```


### 5.2 Compilation Warnings
When compiling MatterSim, you may see deprecation warnings related to PyThread. These warnings are safe to ignore:

```bash
warning: 'int PyThread_set_key_value(int, void*)' is deprecated [-Wdeprecated-declarations]
    PyThread_set_key_value(key, tstate);
```

This warning occurs due to the use of a deprecated Python API function in pybind11, but it does not affect the functionality of the simulator.

### 5.3 MatterSim: Could not open list of scans at "***"

```bash
#modify line 51 in seq2seq/env.py to the correct path
connectivity_dir = '/projects/VLN-Tutorial/Matterport3DSimulator/connectivity'
```

### 5.4 RuntimeError: cuDNN version incompatibility: PyTorch was compiled  against (8, 3, 2) but found runtime version (8, 2, 1)

```bash
## just unset the LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
```

