#### Team name: Violax
#### Challenge name: Object Detection for Autonomous Vehicles using intel oneAPI toolkit
#### Email: samikshasarnaik2001@gmail.com

## Brief of the prototype
The provided code serves as a prototype for object detection in autonomous vehicles using Intel oneDNN, oneDAL, oneMKL, oneTBB, oneVPL, oneCCL, the AI Analytics Toolkit, and SYCL. The prototype demonstrates the following steps:

1. Model Loading: The code loads a pre-trained object detection model by specifying the paths to the model weights, model configuration, and class labels.

2. Image Preprocessing: An input image is loaded and preprocessed to match the expected input dimensions of the model.

3. Computation Graph Creation: The code utilizes Intel oneDNN to create a computation graph. It sets up the graph's input and adds layers such as convolution, batch normalization, activation functions (e.g., ReLU), and others, according to the model configuration.

4. Graph Execution: The computation graph is executed using Intel oneDAL and SYCL for efficient and optimized computations. SYCL provides a way to leverage parallelism and utilize the underlying hardware acceleration (e.g., GPU) for faster processing.

5. Output Processing: The detections from the last layer of the graph are extracted and processed. The code applies a confidence threshold to filter out low-confidence detections. Bounding boxes are generated based on the detected objects' coordinates, and class labels are assigned using the provided class labels file.

6. Visualization: The detected objects are visualized by drawing bounding boxes on the input image. The labels and confidence scores are displayed alongside the bounding boxes.

The prototype showcases the integration of various Intel optimization libraries and frameworks to perform object detection tasks efficiently and harness the power of hardware acceleration.

## Tech Stack of the prototype
The tech stack of the prototype includes the following components and libraries:
1. Intel oneDAL (Data Analytics Library): A library for high-performance data analytics and machine learning. It includes components for data preprocessing, model building, and inference.
2. Intel oneVPL (Video Processing Library): A library for video processing and encoding. It offers APIs for video decoding, encoding, and related operations.
The combination of these components and libraries provides a powerful and optimized tech stack for object detection in autonomous vehicles. It leverages Intel's hardware optimizations, parallel computing capabilities, and deep learning libraries to achieve efficient and high-performance object detection.

## Step by step code execution instruction
1. Install the required dependencies:
   - Intel oneDAL: Install the daal4py package using `pip install daal4py`.
   - Intel oneVPL: Follow the installation instructions from the Intel oneVPL documentation.
   

2. Run train1.py

The code will load the model, preprocess the input image, create the computation graph using oneDNN, execute the graph using oneDAL and SYCL, process the output detections, and draw bounding boxes on the image.

5. The image with bounding boxes will be displayed. Press any key to close the image window.

## What We learnt?
We learnt these things while preparing the prototype:
1. Loading and preprocessing an input image for object detection.
2. Utilizing Intel oneDNN and oneDAL to create a computation graph for object detection.
3. Configuring and connecting different layers in the computation graph.
4. Executing the computation graph using Intel oneDAL and SYCL for efficient computation.
5. Accessing and processing the output detections from the last layer of the graph.
6. Drawing bounding boxes on the input image based on the detected objects.
7. Gaining familiarity with Intel oneMKL, oneTBB, oneVPL, and oneCCL as supporting libraries for optimized computations and parallel processing.
8. Understanding the integration of Intel AI Analytics Toolkit for object detection tasks.

