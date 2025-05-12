# client.py
import requests

url = 'http://localhost:5000/rapp_query'

# 要发送的 JSON payload
payload = {
    'model_name': 'resnet50',
    'batch_size': 4,
    'sm': 40,
    'quota': 50,
}

# 发送 POST 请求
response = requests.post(url, json=payload)

# 输出响应
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(response.json())