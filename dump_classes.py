import joblib
import json
import os

def get_classes(name):
    path = f'ml/models/{name}_encoders.pkl'
    if not os.path.exists(path):
        return {}
    enc = joblib.load(path)
    return {k: v.classes_.tolist() for k, v in enc.items() if hasattr(v, 'classes_')}

data = {
    'career': get_classes('career'),
    'mental_health': get_classes('mental_health'),
    'burnout': get_classes('burnout')
}

with open('all_classes.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Dumped classes to all_classes.json")
