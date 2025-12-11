# 🚀 Serverless Deep Learning Homework - Step-by-Step Instructions

## Machine Learning Zoomcamp 2025 - Module 9

This guide provides complete instructions for deploying the Hair Type Classification model using serverless architecture with AWS Lambda, ONNX, and Docker.

---

## 📚 **Background: What is Module 9 About?**

Module 9 focuses on **Serverless Deep Learning Deployment**:

- **Serverless Computing**: Deploy ML models without managing servers
- **AWS Lambda**: Event-driven, pay-per-request computing service
- **ONNX Format**: Cross-platform model format for efficient deployment
- **Docker Integration**: Package heavy dependencies in containers
- **Modern ML Deployment**: Evolution from TensorFlow Lite to ONNX

### **Key Benefits of Serverless ML:**
✅ **No server management** - Focus on code, not infrastructure  
✅ **Cost-effective** - Pay only for actual usage  
✅ **Auto-scaling** - Handles traffic spikes automatically  
✅ **Event-driven** - Triggered by various AWS events  

---

## 🎯 **Homework Overview**

Deploy the **Straight vs Curly Hair classifier** from Module 8 using:
- Pre-trained ONNX model files
- AWS Lambda with Docker containers  
- Local testing before cloud deployment
- Image preprocessing pipeline from Module 8

**6 Questions to Answer:**
1. ONNX model output node name
2. Target image size for preprocessing
3. First pixel R channel value after preprocessing
4. Model prediction output
5. Docker base image size
6. Model output when run in Docker container

---

## 🛠️ **Prerequisites**

### Required Software:
- **Python 3.8+** with pip
- **Docker** (for Questions 5-6)
- **AWS CLI** (optional, for actual deployment)

### Required Python Packages:
```bash
pip install onnxruntime pillow numpy requests wget
```

### AWS Account (Optional):
- Only needed for actual Lambda deployment
- Not required to complete homework questions

---

## 📋 **Step-by-Step Solution**

### **Step 1: Setup Environment**

```bash
# Create project directory
mkdir serverless-homework
cd serverless-homework

# Install dependencies
pip install onnxruntime pillow numpy requests wget
```

### **Step 2: Download ONNX Model Files**

```bash
# Download using wget (Linux/Mac)
PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
wget ${PREFIX}/hair_classifier_v1.onnx.data
wget ${PREFIX}/hair_classifier_v1.onnx

# Or download using Python
python -c "
import requests

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f'Downloaded {filename}')

prefix = 'https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle'
download_file(f'{prefix}/hair_classifier_v1.onnx.data', 'hair_classifier_v1.onnx.data')
download_file(f'{prefix}/hair_classifier_v1.onnx', 'hair_classifier_v1.onnx')
"
```

### **Step 3: Question 1 - ONNX Model Output Node Name**

```python
import onnxruntime as ort

# Load ONNX model
ort_session = ort.InferenceSession('hair_classifier_v1.onnx')

# Inspect model structure
print("Model Inputs:")
for i, input_meta in enumerate(ort_session.get_inputs()):
    print(f"  {input_meta.name}: {input_meta.shape}")

print("Model Outputs:")
for i, output_meta in enumerate(ort_session.get_outputs()):
    print(f"  {output_meta.name}: {output_meta.shape}")

# Get output node name
output_name = ort_session.get_outputs()[0].name
print(f"Output node name: {output_name}")

# Answer: Compare with options [output, sigmoid, softmax, prediction]
```

### **Step 4: Question 2 - Target Image Size**

From Module 8 homework specifications:
- Model input shape: `(3, 200, 200)` (channels first)
- This means: 3 RGB channels, 200×200 pixels
- Transform used: `transforms.Resize((200, 200))`

**Answer: 200x200**

### **Step 5: Image Preprocessing Functions**

```python
import numpy as np
from PIL import Image
from io import BytesIO
from urllib import request

def download_image(url):
    """Download image from URL"""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    """Prepare image for model input"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    """Complete preprocessing pipeline matching Module 8"""
    # Prepare image
    img = prepare_image(img, target_size=(200, 200))
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Convert to channels-first format (H, W, C) -> (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Apply ImageNet normalization (from Module 8)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
```

### **Step 6: Question 3 - First Pixel R Channel Value**

```python
# Download test image
test_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(test_url)

# Preprocess image
processed_img = preprocess_image(img)

# Get first pixel R channel value
first_pixel_r = processed_img[0, 0, 0, 0]  # batch=0, channel=0 (R), row=0, col=0
print(f"First pixel R channel: {first_pixel_r}")

# Find closest option from [-10.73, -1.073, 1.073, 10.73]
options = [-10.73, -1.073, 1.073, 10.73]
closest = min(options, key=lambda x: abs(x - first_pixel_r))
print(f"Answer: {closest}")
```

### **Step 7: Question 4 - Model Output**

```python
# Run inference
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

outputs = ort_session.run([output_name], {input_name: processed_img})
prediction = outputs[0]

# Extract prediction value
if prediction.shape == (1, 1):
    prediction_value = prediction[0, 0]
elif prediction.shape == (1,):
    prediction_value = prediction[0]
else:
    prediction_value = float(prediction.flatten()[0])

# Apply sigmoid if needed (if not already in [0,1] range)
if not (0 <= prediction_value <= 1):
    prediction_value = 1 / (1 + np.exp(-prediction_value))

print(f"Model output: {prediction_value}")

# Find closest option from [0.09, 0.49, 0.69, 0.89]
options = [0.09, 0.49, 0.69, 0.89]
closest = min(options, key=lambda x: abs(x - prediction_value))
print(f"Answer: {closest}")
```

### **Step 8: Create Lambda Function Code**

Create `lambda_function.py`:

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request
import json

# Load model globally (better performance)
ort_session = ort.InferenceSession('hair_classifier_empty.onnx')
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    img = prepare_image(img, target_size=(200, 200))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array.transpose(2, 0, 1)
    img_array = img_array / 255.0
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_single(url):
    img = download_image(url)
    processed_img = preprocess_image(img)
    outputs = ort_session.run([output_name], {input_name: processed_img})
    prediction = outputs[0]
    
    if prediction.shape == (1, 1):
        prediction_value = prediction[0, 0]
    elif prediction.shape == (1,):
        prediction_value = prediction[0]
    else:
        prediction_value = float(prediction.flatten()[0])
    
    if not (0 <= prediction_value <= 1):
        prediction_value = 1 / (1 + np.exp(-prediction_value))
    
    return float(prediction_value)

def lambda_handler(event, context):
    print("Event:", event)
    url = event.get('url')
    if not url:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'URL parameter is required'})
        }
    
    try:
        prediction = predict_single(url)
        hair_type = "curly" if prediction > 0.5 else "straight"
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction,
                'hair_type': hair_type
            })
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### **Step 9: Question 5 - Docker Base Image Size**

```bash
# Pull the base image
docker pull agrigorev/model-2025-hairstyle:v1

# Check image size
docker images agrigorev/model-2025-hairstyle:v1
```

Look at the SIZE column. Options: [88 Mb, 208 Mb, 608 Mb, 1208 Mb]

**Typical AWS Lambda Python base images are ~600MB**

### **Step 10: Question 6 - Docker Container Testing**

Create `Dockerfile`:

```dockerfile
FROM agrigorev/model-2025-hairstyle:v1

# Install required packages
RUN pip install onnxruntime pillow numpy

# Copy lambda function
COPY lambda_function.py .

# Set command
CMD ["lambda_function.lambda_handler"]
```

Create `test_docker.py`:

```python
import requests
import json

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
test_event = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

try:
    response = requests.post(url, json=test_event, timeout=30)
    result = response.json()
    print("Response:", json.dumps(result, indent=2))
    
    if response.status_code == 200:
        body = json.loads(result['body'])
        prediction = body['prediction']
        print(f"Model prediction: {prediction}")
        
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

**Run the test:**

```bash
# Build image
docker build -t hair-classifier-lambda .

# Run container
docker run -it --rm -p 8080:8080 hair-classifier-lambda

# In another terminal, test
python test_docker.py
```

Look for the prediction value. Options: [-1.0, -0.10, 0.10, 1.0]

---

## 📊 **Expected Results**

Based on typical model behavior and AWS Lambda characteristics:

| Question | Expected Answer | Reasoning |
|----------|----------------|-----------|
| Q1 | `output` or `sigmoid` | Common ONNX output node names |
| Q2 | `200x200` | Module 8 specification |
| Q3 | Around `-1.073` | ImageNet normalization range |
| Q4 | Around `0.49` or `0.69` | Binary classification probability |
| Q5 | `608 Mb` | Typical AWS Lambda base image |
| Q6 | `0.10` | Different model, similar preprocessing |

---

## 🚨 **Common Issues & Solutions**

### **ONNX Runtime Issues:**
```bash
# If onnxruntime fails to install
pip install onnxruntime-cpu  # CPU-only version
```

### **Docker Issues:**
```bash
# If Docker permission denied (Linux)
sudo docker ...

# If port already in use
docker run -it --rm -p 8081:8080 hair-classifier-lambda
# Then use http://localhost:8081 in test
```

### **Image Download Issues:**
```bash
# If SSL certificate issues
pip install --upgrade certifi
```

### **Memory Issues in Lambda:**
- Increase Lambda timeout to 30 seconds
- Increase memory to 1024 MB
- Use smaller batch sizes

---

## 🎯 **Quick Answer Collection**

Use this template for submission:

```
Question 1: [output_node_name]
Question 2: 200x200
Question 3: [first_pixel_value] 
Question 4: [model_output]
Question 5: [docker_image_size]
Question 6: [docker_container_output]
```

---

## 🚀 **Optional: Full AWS Deployment**

### **Deploy to AWS Lambda:**

1. **Create ECR Repository:**
   ```bash
   aws ecr create-repository --repository-name hair-classifier
   ```

2. **Push Docker Image:**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   
   # Tag and push
   docker tag hair-classifier-lambda <account>.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/hair-classifier:latest
   ```

3. **Create Lambda Function:**
   - Go to AWS Console → Lambda
   - Create function → Container image
   - Select your ECR image
   - Increase timeout to 30s
   - Increase memory to 1024 MB

4. **Expose via API Gateway:**
   - Create API Gateway REST API
   - Create resource and method
   - Set integration type to Lambda Function
   - Deploy API

### **Test Deployment:**
```bash
curl -X POST https://your-api-url.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
```

---

## 📚 **Learning Objectives Completed**

✅ **Serverless Architecture** - Deploy ML models without servers  
✅ **ONNX Model Format** - Cross-platform model deployment  
✅ **Docker Integration** - Package dependencies efficiently  
✅ **AWS Lambda** - Event-driven computing service  
✅ **Local Testing** - Test before cloud deployment  
✅ **Image Preprocessing** - Maintain consistency across platforms  

**🎓 You now understand modern serverless ML deployment patterns!**