import pandas as pd
import requests
import os
import json
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--image_directory", type=str)
argument_parser.add_argument("--fitzpatrick_csv", type=str)
argument_parser.add_argument("--output_json", type=str)
args = argument_parser.parse_args()

if not os.path.exists(args.image_directory):
    os.makedirs(args.image_directory)

df = pd.read_csv(args.fitzpatrick_csv)

json_data = []
for index, row in df.iterrows():
    image_filename = f"{row['md5hash']}.jpg"

    try:
        response = requests.get(row['url'])
        response.raise_for_status()  # Raise an error for bad status codes
        with open(os.path.join(args.image_directory, image_filename), 'wb') as image_file:
            image_file.write(response.content)
    except requests.RequestException as e:
        print(f"Error downloading {row['url']}: {e}")
        continue

    json_data.append({
        "id": row['md5hash'],
        "image": image_filename,
        "conversations": [
            {
                "from": "human",
                "value": "Please name the illness which can be seen in the image."
            },
            {
                "from": "gpt",
                "value": row['label']
            }
        ]
    })

with open(args.output_json, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
