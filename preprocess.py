import scispacy
import spacy
import json
from collections import Counter
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

prompt = (
    "You are a helpful medical AI assistant. "
    "You will now recieve a list of terms, some of which are "
    "illnesses. It is your task to pick the illness which was "
    "mentioned most often among the list of words you recieved. "
    "Do not generate any text apart from the one illness mentioned "
    "most often. "
    "List:\n"
)

inference_client = InferenceClient("http://127.0.0.1:8080")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

input_path = "/Projects/marie/weird_skin_spot_challenge/mediqa-m3g-startingkit/mediqa-m3-clinicalnlp2024/train.json"
output_path = "/Projects/marie/weird_skin_spot_challenge/MediQA/mixtral_diagnoses.jsonl"

nlp = spacy.load("en_core_sci_lg")

with open(input_path, "r") as input_file:
    data = json.load(input_file)

with open(output_path, "w") as output_file:
    doc_one = nlp("\n\n".join([response["content_en"] for response in data[0]["responses"]]))
    ents_one = "; ".join([token.text for token in doc_one if token.ent_type_ == "ENTITY"])
    for entry in data[1:]:
        responses = "\n\n".join([response["content_en"] for response in entry["responses"]])
        doc = nlp(responses.lower())

        # Lemmatization, stop word removal
        new_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        doc = nlp(new_text)
        
        # NER
        ents = ";".join([token.text for token in doc if token.ent_type_ == "ENTITY"])

        # Infer illnesses & Diagnoses
        model_input = tokenizer.apply_chat_template(
            [
                {"role" : "user", "content" : (prompt+ents_one)},
                {"role" : "assistant", "content" : "Psoriasis"},
                {"role" : "user", "content" : (prompt+ents)}
            ], tokenize=False)
        print(model_input, "\n\n")
        output = inference_client.text_generation(
                    model_input,
                    max_new_tokens=100,
                    stream=False,
                    details=False
                )
        
        print(output, "\n\n\n")
        output_file.write(json.dumps({"id" : entry["encounter_id"], "prediction" : output})+"\n")
