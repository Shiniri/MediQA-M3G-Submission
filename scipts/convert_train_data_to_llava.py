import json
import argparse
import tqdm
from itertools import product

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_data_path", type=str)
argument_parser.add_argument("--output_data_path", type=str)
argument_parser.add_argument("--language", type=str)
args = argument_parser.parse_args()

processed_data = []
with open(args.input_data_path, "r") as input_file:
    data = json.load(input_file)
    for entry in tqdm.tqdm(data, desc="Converting data for Llava..."):
        for image, response in product(entry["image_ids"], entry["responses"]):
            processed_data.append({
                "id": entry["encounter_id"],
                "image": image,
                "conversations": [
                    {"from": "human", "value": entry[f"query_title_{args.language}"] + "\n\n" + entry[f"query_content_{args.language}"]},
                    {"from": "gpt", "value": response[f"content_{args.language}"]}
                ]
            })

with open(args.output_data_path, "w") as output_file:
    json.dump(processed_data, fp=output_file, ensure_ascii=False)