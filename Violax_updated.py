# -*- coding: utf-8 -*-
import cv2
import numpy as np
from oneapi import dal, dnn, vpl

# Load the MobileNetV2 model from the oneDNN model zoo
model_xml = 'https://github.com/oneapi-src/oneDNN/blob/main/models/public/mobilenet-v2.xml'
model_bin = 'https://github.com/oneapi-src/oneDNN/blob/main/models/public/mobilenet-v2.bin'

# Load the model to the Inference Engine
dnn_model = dnn.Model(model_xml, model_bin)
dnn_model.set_batch_size(1)

# Load the video
cap = cv2.VideoCapture('"C:/Users/samiksha sarnaik/Downloads/violax.mp4')

# Create a VPL pipeline for video decoding
vpl_pipeline = vpl.Pipeline()
vpl_pipeline.set_io_format(vpl.IOFormat.image(b'NV12'), vpl.IOFormat.image(b'RGBP'))

while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to NV12 format for VPL
    nv12_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_YV12)

    # Submit the NV12 frame to VPL pipeline for decoding
    vpl_pipeline.submit(nv12_frame)

    # Decode the frame using VPL pipeline
    rgb_frame = vpl_pipeline.get()

    # Preprocess the input frame
    input_frame = cv2.resize(rgb_frame, (dnn_model.get_input_shape().w, dnn_model.get_input_shape().h))
    input_frame = np.transpose(input_frame, (2, 0, 1)).astype(np.float32)
    input_frame = input_frame / 255.0  # Normalize pixel values to [0, 1]

    # Create a oneDAL tensor from the input frame
    tensor = dal.tensor(input_frame)

    # Run the inference
    result = dnn_model.compute({dnn_model.input_names[0]: tensor})

    # Get the detected objects
    objects = result[dnn_model.output_names[0]]

    # Post-process and visualize the objects
    for obj in objects:
        class_id = int(obj[1])
        confidence = obj[2]
        xmin, ymin, xmax, ymax = map(int, obj[3:7])

        # Draw bounding box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'Class ID: {class_id}, Confidence: {confidence:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
