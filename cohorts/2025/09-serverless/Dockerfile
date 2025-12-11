
FROM agrigorev/model-2025-hairstyle:v1

# Install required Python packages
RUN pip install onnxruntime pillow numpy

# Copy our lambda function
COPY lambda_function.py .

# Set the CMD to our lambda handler
CMD ["lambda_function.lambda_handler"]
