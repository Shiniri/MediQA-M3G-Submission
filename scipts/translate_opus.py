from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import tqdm
import argparse

translator_english = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
tokenizer_english = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator_spanish = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer_spanish = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_data_path", type=str)
argument_parser.add_argument("--en_output_data_path", type=str)
argument_parser.add_argument("--es_output_data_path", type=str)
args = argument_parser.parse_args()

input_data_path = args.input_data_path
with open(input_data_path, "r") as input_file:
    input_data = json.load(input_file)

english_translations = []
spanish_translations = []
for entry in tqdm.tqdm(input_data):

    english_translation = translator_english.generate(
        **tokenizer_english(entry["responses"][0]["content_zh"], return_tensors="pt"),
        max_new_tokens = 100
    )
    decoded_english_translation = [tokenizer_english.decode(token, skip_special_tokens=True) for token in english_translation]
    
    spanish_translation = translator_spanish.generate(
        **tokenizer_spanish(decoded_english_translation[0], return_tensors="pt"),
        max_new_tokens = 100
    )
    decoded_spanish_translation = [tokenizer_spanish.decode(token, skip_special_tokens=True) for token in spanish_translation]

    english_translations.append({
        "encounter_id" : entry["encounter_id"],
        "responses" : [
            {"content_en" : decoded_english_translation[0]}, 
        ]
    })

    spanish_translations.append({
        "encounter_id" : entry["encounter_id"],
        "responses" : [
            {"content_es" : decoded_spanish_translation[0]}, 
        ]
    })

    with open(args.en_output_data_path, "w") as output_file:
        json.dump(english_translations, fp=output_file, ensure_ascii=False)
    with open(args.es_output_data_path, "w") as output_file:
        json.dump(spanish_translations, fp=output_file, ensure_ascii=False)