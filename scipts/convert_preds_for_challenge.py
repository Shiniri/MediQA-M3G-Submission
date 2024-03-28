import json
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_data_path", type=str)
argument_parser.add_argument("--output_data_path", type=str)
argument_parser.add_argument("--language", type=str)
args = argument_parser.parse_args()

converted_entries = []
with open(args.input_data_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        entry = json.loads(line)
        transformed_entry = {
            "encounter_id": entry["question_id"],
            "responses": [{f"content_{args.language}": entry["text"]}]
        }
        converted_entries.append(transformed_entry)

with open(args.output_data_path, 'w', encoding='utf-8') as output_file:
    json.dump(converted_entries, output_file, ensure_ascii=False)