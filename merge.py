import json
from collections import defaultdict

def merge_predictions(input_file, output_file):
    # Initialize a dictionary to hold merged data
    merged_data = defaultdict(lambda: {"ids": [], "count": 0})

    # Read and process each line of the input file
    with open(input_file, 'r') as infile:
        for line in infile:
            print(line)
            data = json.loads(line)
            prediction_key = data["prediction"].lower()  # Normalize the prediction to ignore case

            # Update the list of ids and the count for this prediction
            merged_data[prediction_key]["ids"].append(data["id"])
            merged_data[prediction_key]["count"] += 1

    # Write the merged results to the output file
    with open(output_file, 'w') as outfile:
        for prediction, info in merged_data.items():
            # Create a new dictionary for each prediction
            output_data = {
                "prediction": prediction,
                "ids": info["ids"],
                "count": info["count"]
            }
            # Write this dictionary as a JSON line in the output file
            json_line = json.dumps(output_data)
            outfile.write(json_line + '\n')

# Example usage
input_file = '/Projects/marie/weird_skin_spot_challenge/MediQA/mixtral_diagnoses_handpicked.jsonl'
output_file = '/Projects/marie/weird_skin_spot_challenge/MediQA/merged.jsonl'
merge_predictions(input_file, output_file)
