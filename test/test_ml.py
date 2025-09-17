import requests

url = "http://localhost:8000/predict"
file_path = "/home/semaistek/guardvision_ml/project/images/test2.jpg" 

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.json())