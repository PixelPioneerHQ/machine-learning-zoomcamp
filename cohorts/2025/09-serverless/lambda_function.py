
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request
import json

# Load the model globally (outside the handler for better performance)
ort_session = ort.InferenceSession('hair_classifier_empty.onnx')
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

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
    """Complete preprocessing pipeline"""
    # Prepare image
    img = prepare_image(img, target_size=(200, 200))

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Convert to channels-first format (H, W, C) -> (C, H, W)
    img_array = img_array.transpose(2, 0, 1)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_single(url):
    """Predict hair type for a single image URL"""
    # Download and preprocess image
    img = download_image(url)
    processed_img = preprocess_image(img)

    # Run inference
    outputs = ort_session.run([output_name], {input_name: processed_img})
    prediction = outputs[0]

    # Extract prediction value
    if prediction.shape == (1, 1):
        prediction_value = prediction[0, 0]
    elif prediction.shape == (1,):
        prediction_value = prediction[0]
    else:
        prediction_value = float(prediction.flatten()[0])

    # Apply sigmoid if needed (if output is not in [0,1] range)
    if not (0 <= prediction_value <= 1):
        prediction_value = 1 / (1 + np.exp(-prediction_value))

    return float(prediction_value)

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    print("Event:", event)

    # Extract URL from event
    url = event.get('url')
    if not url:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'URL parameter is required'})
        }

    try:
        # Make prediction
        prediction = predict_single(url)

        # Determine hair type (assuming > 0.5 means curly)
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
