import scispacy
import spacy
import json
from collections import Counter
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

prompt = (
    "You are a helpful medical AI assistant. "
    "You will now recieve a list of words. "
    "Decide which of said words can be considered an illness. "
    "Answer by writing out the words which can be considered an illness. "
    "Do not generate any other text apart from that. "
    "LIST: \n"
)

inference_client = InferenceClient("http://127.0.0.1:8080")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

input_path = "/Projects/marie/weird_skin_spot_challenge/mediqa-m3g-startingkit/mediqa-m3-clinicalnlp2024/train.json"
output_path = "/Projects/marie/weird_skin_spot_challenge/MediQA/mixtral_diagnoses.jsonl"

nlp = spacy.load("en_core_sci_lg")

with open(input_path, "r") as input_file:
    data = json.load(input_file)

with open(output_path, "w") as output_file:
    for entry in data:
        responses = "\n\n".join([response["content_en"] for response in entry["responses"]])
        doc = nlp(responses.lower())

        # Lemmatization, stop word removal
        new_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        doc = nlp(new_text)
        
        # NER
        ents = ";".join([token.text for token in doc if token.ent_type_ == "ENTITY"])

        # Infer illnesses & Diagnoses
        model_input = tokenizer.apply_chat_template([{"role" : "user", "content" : (prompt+ents)}], tokenize=False)
        output = inference_client.text_generation(
                    model_input,
                    max_new_tokens=100,
                    stream=False,
                    details=False
                )
        
        print(output, "\n\n\n")
        output_file.write(json.dumps({"id" : entry["encounter_id"], "prediction" : output})+"\n")
