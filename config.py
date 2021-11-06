import json


def load_reslut():
    with open('./result.json', 'r+') as f:
        return json.load(f)


result: {} = load_reslut()
