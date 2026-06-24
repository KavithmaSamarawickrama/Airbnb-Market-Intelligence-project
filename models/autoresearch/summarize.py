import json

with open("models/autoresearch/history.json") as f:
    data = json.load(f)

print(json.dumps(data[-1], indent=2))
