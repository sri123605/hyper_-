import urllib.request
import urllib.parse
import re

data = urllib.parse.urlencode({
    'gender': 'Male', 
    'age': '18-34', 
    'family_history': 'Yes', 
    'medication': 'Yes', 
    'symptom_severity': 'Mild', 
    'shortness_of_breath': 'Yes', 
    'visual_changes': 'Yes', 
    'nosebleeds': 'Yes', 
    'sys_bp': '<120', 
    'dia_bp': '<80', 
    'diet_control': 'Yes'
}).encode('utf-8')

try:
    req = urllib.request.Request('http://127.0.0.1:5000/', data=data)
    html = urllib.request.urlopen(req).read().decode('utf-8')
    match = re.search(r'<div class="alert alert-danger">\s*(.*?)\s*</div>', html, re.DOTALL)
    if match:
        print('ERROR FOUND:', match.group(1).strip())
    else:
        print('SUCCESS! No error div')
except Exception as e:
    print("Request failed:", e)
