
import requests
import json

# Test the lambda function running in Docker
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# Test image URL
test_event = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

try:
    response = requests.post(url, json=test_event, timeout=30)
    result = response.json()
    print("Response status:", response.status_code)
    print("Response body:", json.dumps(result, indent=2))

    if response.status_code == 200:
        body = json.loads(result['body'])
        prediction = body['prediction']
        print(f"
Model prediction: {prediction}")
        print(f"Hair type: {body['hair_type']}")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print("Make sure Docker container is running on port 8080")
