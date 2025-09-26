import json
import base64

with open("translate-jaari-065fa764be8a.json", "r") as f:
    data = json.load(f)

# On encode le JSON en base64
b64 = base64.b64encode(json.dumps(data).encode("utf-8")).decode("utf-8")
print(b64)