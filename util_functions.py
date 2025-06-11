import lmstudio as lms
import base64
from global_variables import VISION_MODELS
from huggingface_hub import InferenceClient
from google import genai
from api_keys import HUGGINGFACE_HUB_API_KEY, GEMINI_API_KEY
from collections import Counter
import regex as re
from rapidfuzz import process, fuzz
# from ultralytics import YOLO


# def load_vision_model(vision_model_name):
#     try:
#         print(f"Loading vision model {vision_model_name}...")
#         vision_model = lms.llm(vision_model_name)
#     except Exception as e:
#         print(f"Error loading model: {e}\n")
#         return None
#     else:
#         print(f"Model {vision_model_name} loaded successfully!\n")
#         return vision_model

def tokenize_medicine_name(name):
    r = r"(?<!\d)[.](?!\d)|[()\[\]{}|&,;'\"\\/\-_%]|(?=\+)"
    r2 = re.compile(r'^(\.\d+)|(\d+(?:\.\d+)?)([a-z]+)$')
    abbrs = {
        "tab": "tablet",
        "syp": "syrup",
        "syr": "syrup",
        "dil": "dilute",
        "lot": "lotion",
        "cr": "cream",
        "crm": "cream",
        "cap": "capsule",
        "inj": "injection",
        "pulv": "powder"
    }
    units = {"ml", "mcg", "mg", "au", "cu", "cst", "cm", "each", "g", "gm", "iu", "k", "kiu", "kg", "l", "lf", "lac", "pfu", "ppm", "ply", "w", "units", "xl"}
    
    if name:
        words = name.split()
        finalset = set()
        for word in words:
            parts = re.split(r, word)
            for part in parts:
                part = part.strip().lower()
                if part:
                    match = r2.match(part)
                    if match:
                        # print(f"Matched {part}")
                        groups = match.groups()
                        for group in groups:
                            # print(f"\tGroup: {group}")
                            if group and group not in units:
                                # if group in abbrs:
                                #     group = abbrs[group]
                                finalset.add(group)
                    else:
                        if part not in units:
                            if part in abbrs:
                                part = abbrs[part]
                            finalset.add(part)        

        finalstr = " ".join(sorted(list(finalset)))
        return finalstr

# def generate_ngrams(word, n = 3):
#     padded = f"{'~'*(n-1)}{word}{'~'*(n-1)}"  # pad with special chars to capture edges
#     return [padded[i : i+n] for i in range(len(padded) - n + 1)]

# def ngram_score(query_token, candidate_token, n = 3):
#     q_ngrams = Counter(generate_ngrams(query_token, n))
#     c_ngrams = Counter(generate_ngrams(candidate_token, n))
#     common = sum((q_ngrams & c_ngrams).values())
#     total = max(sum(q_ngrams.values()), sum(c_ngrams.values()))
#     return common / total if total else 0.0

# def ngram_spell_correct(query, tokenized_candidates, raw_candidates, top_n = 3, n = 3):
#     query_tokens = tokenize_medicine_name(query).split()
#     ranked = []

#     for idx, tokenized_str in enumerate(tokenized_candidates):
#         candidate_tokens = tokenized_str.split()
        
#         # Score = average max ngram score per query token
#         scores = []
#         for qt in query_tokens:
#             token_scores = [ngram_score(qt, ct, n) for ct in candidate_tokens]
#             if token_scores:
#                 scores.append(max(token_scores))
        
#         if scores:
#             avg_score = sum(scores) / len(scores)
#             ranked.append((raw_candidates[idx], avg_score))
    
#     ranked.sort(key = lambda x: x[1], reverse = True)
#     return ranked[: top_n]

def token_similarity(query_tokens, candidate_tokens):
    total_score = 0
    for q in query_tokens:
        best_score = max((fuzz.WRatio(q, c) for c in candidate_tokens), default = 0)
        total_score += best_score
    return total_score / len(query_tokens)

def search_medicine(query, search_strings, medicine_names, top_k = 25):
    query = tokenize_medicine_name(query)
    matches = process.extract(query, search_strings, scorer = fuzz.WRatio, limit = 30)
    query_tokens = query.split()
    ranked = []
    for match_str, _score, index in matches:
        candidate_tokens = match_str.split()
        score = token_similarity(query_tokens, candidate_tokens)
        actual_name = medicine_names[index]
        ranked.append((actual_name, score))

    ranked.sort(key = lambda x: x[1], reverse = True)
    return ranked[: top_k]

def refine_response(response, reasoning_model_identifier, df):
    try:
        print("Refining response...")
        reasoning_model = lms.llm(reasoning_model_identifier)
        suggestions = ["SUGGESTIONS"]

        for line in response.split("\n"):
            if line.startswith("MEDICINE"):
                query = line.split("MEDICINE:")[-1]
                tokenized_query = tokenize_medicine_name(query)
                suggested_names = " ".join([f"'{name}'" for (name, score) in search_medicine(query, df.search_string, df.medicine_name)])
                print(query)
                print(suggested_names)
                prompt = f"Here is a reference medicine name string: '{query}'. And here are some suggested medicine names: {suggested_names}. Identify the suggested medicine name most similar to the reference one and return the exact name. Do not return anything except the correct suggested medicine name."
                suggestion = reasoning_model.respond(prompt)
                suggestion = str(suggestion).strip().replace("'", "")
                suggestions.append(suggestion)
            elif line.startswith("INSTRUCTION"):
                suggestions.append(line.split("INSTRUCTION:")[-1])
        
        # refined_response = response + "\n\n" + "\n".join(suggestions)
        refined_response = "\n".join(suggestions)
        
        # prompt = f"Correct the incorrect lines in the following text enclosed within backticks. Preserve the line prefix (such as 'MEDICINE:', 'INSTRUCTION:', etc.) and the line breaks. The corrections are supposed to be made in the tablet, syrup or injection names as well as the dosage suggestions:\n\n`{response}`"
        # refined_response = reasoning_model.respond(prompt)
        # refined_response = str(response)
    except Exception as e:
        print(f"Error refining response: {e}\n")
        return None
    else:
        return refined_response
    
def process_image(image_path):
    # image_path = rf"IITRPR Handwritten Prescriptions\{i}.jpg"
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{base64_image}"
    
    return image_url

def analyze_prescription(image, vision_model_name, reasoning_model_identifier, df):
    try:
        print("Analyzing prescription...")
        image.save("temp.jpg")
        # image_url = process_image("temp.jpg")
        prompt = "This is a handwritten medical prescription, treat it accordingly. For each line, follow these instructions: \nFirst check if it is a medical information line or something else. If it is empty or doesn't contain any visible information, return 'NOTHING'. If it is a signature, ignore it and return 'SIGNATURE'. If it is a date, return only the date in DD-MM-YYYY format after 'DATE:'. If it is an actual medical information line, write only whatever is clear - medicine name, dosage, etc after 'MEDICINE:'. If it is any instruction related to taking the medicine, such as 'x5d', then write it as it is after 'INSTRUCTION:'. End each line with a newline (so that the next line starts on a new line). Strictly NO UNWANTED TEXT IS ALLOWED. Only write what is visible."
        model = VISION_MODELS[vision_model_name]

        if vision_model_name == "Gemini 2.5 Flash":
            client = genai.Client(api_key = GEMINI_API_KEY)
            image_url = client.files.upload(file = "temp.jpg")
            messages = [
                image_url,
                prompt
            ]
            response = client.models.generate_content(
                model = model,
                contents = messages,
                config = genai.types.GenerateContentConfig(
                    temperature= 0.6,
                    seed = 0
                )
            )
            response = response.text
        elif vision_model_name in ["Llama 4 Scout", "Llama 4 Maverick"]:
            client = InferenceClient(
                provider = "sambanova",
                api_key = HUGGINGFACE_HUB_API_KEY
            )
            image_url = process_image("temp.jpg")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            completion = client.chat.completions.create(
                model = model,
                messages = messages,
                max_tokens = 500,
            )
            response = completion.choices[0].message.content

        response = refine_response(str(response), reasoning_model_identifier, df)
    except Exception as e:
        print(f"Error analyzing prescription: {e}\n")
        return None
    else:
        print("Prescription analyzed successfully!\n")
        return response
    