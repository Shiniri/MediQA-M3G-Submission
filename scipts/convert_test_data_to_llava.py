import json
import tqdm
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_data_path", type=str)
argument_parser.add_argument("--output_data_path", type=str)
argument_parser.add_argument("--language", type=str)
args = argument_parser.parse_args()

with open(args.input_data_path, "r") as input_open, open(args.output_data_path, "w") as output_open:
    content = json.loads(input_open.read())
    finished_data = []
    for entry in tqdm.tqdm(content, desc="Converting test data for Llava..."):
        images = entry["image_ids"]
        user_prompt = entry[f"query_title_{args.language}"] + "\n\n" + entry[f"query_content_{args.language}"]

        finished_data.append({
            "question_id" : entry["encounter_id"],
            "image" : entry["image_ids"][0],
            "text" : user_prompt
        })

    for entry in finished_data:
        output_open.write(json.dumps(entry, ensure_ascii=False)+"\n")