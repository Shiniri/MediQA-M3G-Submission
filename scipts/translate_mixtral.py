from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import tqdm
import argparse

sys_prompt_chinese = """
You are a translator who produces accurate and complete translations from Chinese to English.
You will now recieve the text from a Chinese forum post on a dermatological topic.
It is your task to produce an accurate English translation of that Chinese post.
Remember to generate nothing but the translation. Do not add any additional information.

=== Begin Post ===
{query}
=== End Post ===
"""

sys_prompt_spanish = """
Eres un traductor que produce traducciones precisas y completas del chino al español.
Ahora recibirás el texto de una publicación en un foro chino sobre un tema dermatológico.
Tu tarea es producir una traducción precisa al español de esa publicación en chino.
Recuerda generar únicamente la traducción. No agregues información adicional.

=== Inicio de la publicación ===
{query}
=== Fin de la publicación ===
"""

translator = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--input_data_path", type=str)
argument_parser.add_argument("--output_data_path", type=str)
args = argument_parser.parse_args()

input_data_path = args.input_data_path
output_data_path = args.output_data_path
with open(input_data_path, "r") as input_file:
    input_data = json.load(input_file)

translations = []
for entry in tqdm.tqdm(input_data):
    english_chat = tokenizer.apply_chat_template(
        [
            {"role" : "user", "content" : sys_prompt_chinese.format(query = "我认为这是神经性皮炎")},
            {"role" : "assistant", "content" : "I think this is Neurodermatitis."},
            {"role" : "user", "content" : sys_prompt_chinese.format(query = "这是带状疱疹，有针对这种情况的抗病毒眼药水。")},
            {"role" : "assistant", "content" : "This is herpes zoster, there is antiviral eye drops for this."},
            {"role" : "user", "content" : sys_prompt_chinese.format(query = entry["responses"][0]["content_zh"])}
        ],
        return_tensors = "pt"
    ).to("cuda")

    english_translation = translator.generate(
        english_chat,
        max_new_tokens = 100
    )
    english_translation = english_translation[:, english_chat.shape[1]:]

    spanish_chat = tokenizer.apply_chat_template(
        [
            {"role" : "user", "content" : sys_prompt_spanish.format(query = "我认为这是神经性皮炎")},
            {"role" : "assistant", "content" : "Creo que esto es Neurodermatitis."},
            {"role" : "user", "content" : sys_prompt_spanish.format(query = "这是带状疱疹，有针对这种情况的抗病毒眼药水")},
            {"role" : "assistant", "content" : "Esto es herpes zóster, hay gotas oftálmicas antivirales para esto."},
            {"role" : "user", "content" : sys_prompt_spanish.format(query = entry["responses"][0]["content_zh"])}
        ],
        return_tensors = "pt"
    ).to("cuda")

    spanish_translation = translator.generate(
        spanish_chat,
        max_new_tokens = 100
    )
    spanish_translation = spanish_translation[:, spanish_chat.shape[1]:]

    decoded_english_translation = [tokenizer.decode(token, skip_special_tokens=True) for token in english_translation]
    decoded_spanish_translation = [tokenizer.decode(token, skip_special_tokens=True) for token in spanish_translation]

    cleaned_english_translation = re.sub(r"(\(.*\))|(Note:.*)|(\n.*)", "", decoded_english_translation[0])
    cleaned_spanish_translation = re.sub(r"(\(.*\))|(Nota:.*)|(\n.*)", "", decoded_spanish_translation[0])
    print("\n\n"+cleaned_english_translation+"\n\n")
    print("\n\n"+cleaned_spanish_translation+"\n\n")

    translations.append({
        "encounter_id" : entry["encounter_id"],
        "responses" : [
            {"content_zh" : entry["responses"][0]["content_zh"]},
            {"content_en" : cleaned_english_translation},
            {"content_es" : cleaned_spanish_translation}   
        ]
    })

with open(output_data_path, "w") as output_file:
    json.dump(translations, fp=output_file, ensure_ascii=False)