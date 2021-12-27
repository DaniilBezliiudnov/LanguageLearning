import json
with open("/data/1.json", 'r') as f:
    datastore = json.load(f)
    for item in datastore:
        sentence = item['headline'].lower()
        label = item['is_sarcastic']
        link = item['article_link']