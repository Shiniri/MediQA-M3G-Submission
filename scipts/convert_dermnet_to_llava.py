import argparse
import os
import json

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_data_path", type=str)
argument_parser.add_argument("--output_data_path", type=str)
args = argument_parser.parse_args()

sys_prompt = """
What illness can be seen in this image?
"""

parsed_data = []
id = 1
for class_dir in os.listdir(args.input_data_path):
    image_dir = os.path.join(args.input_data_path, class_dir)
    for image in os.listdir(image_dir):
        print(image)
        parsed_data.append({
            "id" : id,
            "image" : os.path.join(class_dir, image),
            "conversations" : [
                {"from" : "human", "value" : sys_prompt},
                {"from" : "gpt", "value" : image.replace(".jpg", "")}
            ]
        })
        id += 1

with open(args.output_data_path, "w") as output_file:
    json.dump(parsed_data, fp=output_file)