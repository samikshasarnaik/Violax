# Team name: Violax
# Challenge name: Object Detection for Autonomous Vehicles using intel oneAPI toolkit
# Leader Name: Samiksha Sarnaik
Email: samikshasarnaik2001@gmail.com

# Brief of the prototype
The provided code serves as a prototype for object detection in autonomous vehicles using Intel oneDNN, oneDAL, oneMKL, oneTBB, oneVPL, oneCCL, the AI Analytics Toolkit, and SYCL. The prototype demonstrates the following steps:

1. Model Loading: The code loads a pre-trained object detection model by specifying the paths to the model weights, model configuration, and class labels.

2. Image Preprocessing: An input image is loaded and preprocessed to match the expected input dimensions of the model.

3. Computation Graph Creation: The code utilizes Intel oneDNN to create a computation graph. It sets up the graph's input and adds layers such as convolution, batch normalization, activation functions (e.g., ReLU), and others, according to the model configuration.

4. Graph Execution: The computation graph is executed using Intel oneDAL and SYCL for efficient and optimized computations. SYCL provides a way to leverage parallelism and utilize the underlying hardware acceleration (e.g., GPU) for faster processing.

5. Output Processing: The detections from the last layer of the graph are extracted and processed. The code applies a confidence threshold to filter out low-confidence detections. Bounding boxes are generated based on the detected objects' coordinates, and class labels are assigned using the provided class labels file.

6. Visualization: The detected objects are visualized by drawing bounding boxes on the input image. The labels and confidence scores are displayed alongside the bounding boxes.

The prototype showcases the integration of various Intel optimization libraries and frameworks to perform object detection tasks efficiently and harness the power of hardware acceleration. It serves as a starting point for developing more advanced and customized object detection solutions for autonomous vehicles.

# Tech Stack of the prototype
The tech stack of the prototype includes the following components and libraries:

1. Intel oneDNN (Deep Neural Network Library): A performance-optimized library for deep learning inference. It provides efficient implementations of neural network primitives and supports various hardware accelerators.

2. Intel oneDAL (Data Analytics Library): A library for high-performance data analytics and machine learning. It includes components for data preprocessing, model building, and inference.

3. Intel oneMKL (Math Kernel Library): A highly optimized mathematical library that provides functions for linear algebra, FFT (Fast Fourier Transform), and other mathematical operations.

4. Intel oneTBB (Threading Building Blocks): A library for parallel programming that enables efficient task-based parallelism. It provides constructs for concurrent execution and synchronization.

5. Intel oneVPL (Video Processing Library): A library for video processing and encoding. It offers APIs for video decoding, encoding, and related operations.

6. Intel oneCCL (Collective Communications Library): A library for efficient collective communication and distributed computing. It enables parallel execution across multiple compute nodes.

7. AI Analytics Toolkit: A suite of Intel tools and libraries for AI and analytics workloads. It includes components such as oneDNN, oneDAL, and other optimized libraries.

8. SYCL (Single-source Heterogeneous Programming): A programming model that allows developers to write code that can be executed on various hardware accelerators, including CPUs, GPUs, and FPGAs.

The combination of these components and libraries provides a powerful and optimized tech stack for object detection in autonomous vehicles. It leverages Intel's hardware optimizations, parallel computing capabilities, and deep learning libraries to achieve efficient and high-performance object detection.

# Step by step code execution instruction
1. Install the required dependencies:
   - Intel oneDNN: Follow the installation instructions from the Intel oneDNN documentation.
   - Intel oneDAL: Install the daal4py package using `pip install daal4py`.
   - Intel oneMKL: Install the mkl package using `pip install mkl`.
   - Intel oneTBB: Install the tbb package using `pip install tbb`.
   - Intel oneVPL: Follow the installation instructions from the Intel oneVPL documentation.
   - Intel oneCCL: Follow the installation instructions from the Intel oneCCL documentation.
   - AI Analytics Toolkit: Follow the installation instructions from the AI Analytics Toolkit documentation.
   - SYCL: Install the daal4py.oneapi package using `pip install daal4py.oneapi`.

2. Replace `"path/to/model.weights"` with the actual path to your model's weights file, `"path/to/model.config"` with the path to the model's configuration file, and `"path/to/model.classes"` with the path to the file containing class labels.

3. Replace `"path/to/image.jpg"` with the actual path to the input image.

4. Run the code using a Python interpreter.
   violax.py
The code will load the model, preprocess the input image, create the computation graph using oneDNN, execute the graph using oneDAL and SYCL, process the output detections, and draw bounding boxes on the image.

5. The image with bounding boxes will be displayed. Press any key to close the image window.

# What We learnt?
We learnt these things while preparing the prototype:
1. Loading and preprocessing an input image for object detection.
2. Utilizing Intel oneDNN and oneDAL to create a computation graph for object detection.
3. Configuring and connecting different layers in the computation graph.
4. Executing the computation graph using Intel oneDAL and SYCL for efficient computation.
5. Accessing and processing the output detections from the last layer of the graph.
6. Drawing bounding boxes on the input image based on the detected objects.
7. Gaining familiarity with Intel oneMKL, oneTBB, oneVPL, and oneCCL as supporting libraries for optimized computations and parallel processing.
8. Understanding the integration of Intel AI Analytics Toolkit for object detection tasks.

