import requests
import sys
import os

# Move to backend to import auth_utils
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from app.utils.auth_utils import create_access_token

# Generate token for prashant
token = create_access_token(data={"sub": "prashant@gmail.com"})

url = "http://127.0.0.1:8000/predict/career"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
payload = {
    "University": "Aligarh Muslim University",
    "Course": "BTech",
    "Department": "Engineering",
    "Branch": "Software",
    "Semester": 6,
    "CGPA": 8.5,
    "Interest": "Job",
    "Mode_of_Study": "Regular",
    "Institution_Type": "Government",
    "Location": "Urban",
    "Medium_of_Education": "English"
}

try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
