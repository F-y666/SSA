import json
import re
import torch
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from collections import defaultdict, Counter
import time
import os

from openai import OpenAI, APIConnectionError, RateLimitError, Timeout
# from langchain_community.llms import OpenAI as LangOpenAI
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = ''

MODEL_DIR = './Model/Llama-3.1-8B-Instruct'
DOCTOR_RESPONSES_PATH = "./Information/Doctor-responses.json"
OUTPUT_DIR = "./Huffman/chatdoctor"
SYNONYM_REPLACEMENT_PATH = os.path.join(OUTPUT_DIR, "synonym_replacements.json")

MIN_CHAR_LEN = 50
MAX_CHAR_LEN = 5000
SAMPLE_SIZE = 50
RETRY_LIMIT = 2

SYNONYM_PROMPT = """
You are a language expert. Given the following sentence, identify only content-bearing tokens (such as nouns, verbs, adjectives, or domain-specific terms) that can be replaced with synonyms **without changing the core meaning, tone, or grammatical correctness** of the sentence.

For each valid token, output exactly two lines, strictly in this format:
1. The original token (a single word or short phrase) surrounded by double asterisks, e.g., **token**
2. A JSON list of exactly 3 distinct synonyms (excluding any form of the original), e.g., ["syn1", "syn2", "syn3"]

STRICT FORMAT INSTRUCTIONS:
- Output only pairs of lines in the format above. No explanation, headings, bullet points, or additional comments.
- The synonym list MUST be valid JSON: square brackets, double quotes, comma-separated, no trailing commas.
- Do NOT include the original token or any of its inflected, derived, or morphological forms in the synonym list (case-insensitive).
- Do NOT output any content other than the required token-synonym pairs. No blank lines or summary statements.

Sentence: "{sentence}"

Output only as specified above.
"""


def ensure_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def normalize_token(word):

    w = word.lower().strip()
    w = re.sub(r"[\s\-\_]+", "", w)
    w = re.sub(r"[.,;:!?()\"']", "", w)
    return w

def extract_synonym_format(text):

    entries = []
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines) - 1:
        token_line = lines[i].strip()
        syn_line = lines[i + 1].strip()
        token_match = re.match(r"(?:\d+\.\s*)?\*\*\s*(.+?)\s*\*\*", token_line)
        if not token_match:
            i += 1
            continue
        token = token_match.group(1).strip()
        syn_match = re.search(r"\[(.*?)\]", syn_line)
        if not syn_match:
            i += 2
            continue
        try:
            arr_txt = "[" + syn_match.group(1).strip() + "]"
            synonyms = json.loads(arr_txt)
            if not isinstance(synonyms, list):
                raise ValueError
            token_norm = normalize_token(token)
            cleaned_synonyms = []
            for s in synonyms:
                s_norm = normalize_token(s)
                if s_norm and s_norm != token_norm:
                    cleaned_synonyms.append(s.strip())
            if len(cleaned_synonyms) >= 2:
                entries.append({
                    "token": token.lower().strip(),
                    "synonyms": [s.lower().strip() for s in cleaned_synonyms]
                })
        except Exception:
            pass
        i += 2
    return entries

def normalize(text):
    return text.lower().strip()

def safe_invoke_openai(llm, prompt, retries=3, delay=2):
    for i in range(retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            time.sleep(delay)

    return "[NO_SYNONYMS]"


def query_synonyms_local(sentence, pipe, tokenizer, retries=RETRY_LIMIT):
    prompt = SYNONYM_PROMPT.format(sentence=sentence)
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            response = pipe(
                prompt,
                do_sample=True,
                max_new_tokens=256,
                temperature=0.4,
                num_return_sequences=1,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            result_text = response[0]["generated_text"].strip()
            entries = extract_synonym_format(result_text)
            if entries:
                return entries
            else:
                raise ValueError("No valid synonym blocks extracted.")
        except Exception as e:
            last_exception = e
            print(f"[!] Local extraction failed (Attempt {attempt}/{retries}): {e}")
            time.sleep(1)
    print(f"[!] All local extraction attempts failed for sentence. Error: {last_exception}")
    return []

def query_synonyms_openai(sentence, llm, retries=RETRY_LIMIT):
    prompt = SYNONYM_PROMPT.format(sentence=sentence)
    last_exception = None
    for attempt in range(1, retries + 1):
        try:

            response = safe_invoke_openai(llm, prompt)
            result_text = response.strip()
            entries = extract_synonym_format(result_text)
            if entries:
                return entries
            else:
                raise ValueError("No valid synonym blocks extracted.")
        except Exception as e:
            last_exception = e
            print(f"[!] OpenAI extraction failed (Attempt {attempt}/{retries}): {e}")
            time.sleep(1)
    print(f"[!] All OpenAI extraction attempts failed for sentence. Error: {last_exception}")
    return []

def load_doctor_responses(path, min_len, max_len, sample_size):
    with open(path, "r", encoding="utf-8") as f:
        all_responses = json.load(f)
    filtered = [r for r in all_responses if min_len <= len(r) <= max_len]
    if len(filtered) == 0:
        raise RuntimeError(f"No sentences found with length in [{min_len}, {max_len}]")
    if len(filtered) < sample_size:
        print(f"[!] Warning: Not enough responses ({len(filtered)}) for requested sample_size ({sample_size}), using all.")
        return filtered

    return random.sample(filtered, sample_size)

import heapq

def build_huffman_code(counter):
    heap = [[freq, [syn, ""]] for syn, freq in counter.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        syn = heap[0][1][0]
        return {syn: "0"}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    result = {}
    for syn, code in heap[0][1:]:
        result[syn] = code
    return result

def build_all_huffman_codes(filtered_token_synonym_freq):
    token_huffman_codebook = {}
    for token, counter in filtered_token_synonym_freq.items():
        codebook = build_huffman_code(counter)
        token_huffman_codebook[token] = codebook
    return token_huffman_codebook


def main(model_type='openai', codebook_sizes=[50, 100, 200, 500]):
    ensure_dir(OUTPUT_DIR)

    if os.path.exists(SYNONYM_REPLACEMENT_PATH):
        print(f"🚦 Detected existing synonym replacements: {SYNONYM_REPLACEMENT_PATH}")
        with open(SYNONYM_REPLACEMENT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)

    else:
        results = []

    processed_sentences = set(item["sentence"] for item in results)

    print("📚 Loading and sampling doctor responses ...")
    responses = load_doctor_responses(
        DOCTOR_RESPONSES_PATH,
        MIN_CHAR_LEN,
        MAX_CHAR_LEN,
        SAMPLE_SIZE
    )

    remaining_responses = [s for s in responses if s not in processed_sentences]
    print(f"✅ {len(remaining_responses)} sentences left to process (skipped {len(processed_sentences)}).")

    if remaining_responses:

        if model_type == 'local':
            print(f"🧬 Loading local model from {MODEL_DIR} ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
            pipe = transformers.pipeline(
                "text-generation",
                model=MODEL_DIR,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto"
            )
            query_func = lambda s: query_synonyms_local(s, pipe, tokenizer)
        elif model_type == 'openai':
            print("🔗 Using OpenAI GPT API ...")
            llm = ChatOpenAI(
                model='o1-mini',
                temperature=0.6,
                request_timeout=100
            )
            query_func = lambda s: query_synonyms_openai(s, llm)
        else:
            raise ValueError("Invalid model_type, must be 'local' or 'openai'.")

        for i, sentence in enumerate(tqdm(remaining_responses, desc="Processing")):
            synonym_data = query_func(sentence)
            if not synonym_data:
                tqdm.write(f"[!] Skipped sentence #{i} due to extraction failure.")
                continue

            results.append({
                "sentence": sentence,
                "replacements": synonym_data
            })

            with open(SYNONYM_REPLACEMENT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ Synonym replacement complete. Saved to {SYNONYM_REPLACEMENT_PATH}")

    token_synonym_freq = defaultdict(Counter)
    for entry in results:
        for repl in entry.get("replacements", []):
            token = normalize(repl["token"])
            for syn in repl.get("synonyms", []):
                syn_norm = normalize(syn)
                if syn_norm and syn_norm != token:
                    token_synonym_freq[token][syn_norm] += 1

    token_total_freq = {token: sum(syns.values()) for token, syns in token_synonym_freq.items()}
    sorted_tokens = sorted(token_total_freq.items(), key=lambda x: x[1], reverse=True)

    for top_k in codebook_sizes:
        top_token_set = set(t[0] for t in sorted_tokens[:top_k])
        filtered_token_synonym_freq = {
            token: dict(token_synonym_freq[token])
            for token in top_token_set
            if len(token_synonym_freq[token]) >= 1
        }
        print(f"📊 [Top {top_k}] token count: {len(filtered_token_synonym_freq)}")

        huffman_codebook_path = os.path.join(OUTPUT_DIR, f"huffman_codebook_top{top_k}.json")
        token_synonym_path = os.path.join(OUTPUT_DIR, f"cleaned_token_synonyms_top{top_k}.json")
        token_freq_path = os.path.join(OUTPUT_DIR, f"token_frequency_top{top_k}.json")

        token_huffman_codebook = build_all_huffman_codes(filtered_token_synonym_freq)
        with open(huffman_codebook_path, "w", encoding="utf-8") as f:
            json.dump(token_huffman_codebook, f, ensure_ascii=False, indent=2)
        with open(token_synonym_path, "w", encoding="utf-8") as f:
            json.dump(filtered_token_synonym_freq, f, indent=2, ensure_ascii=False)
        with open(token_freq_path, "w", encoding="utf-8") as f:
            json.dump(sorted_tokens[:top_k], f, indent=2, ensure_ascii=False)
        print(f"✅ Huffman & synonym files saved for top {top_k}")

    print(f"✅ All codebooks done: {codebook_sizes}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['local', 'openai'], default='openai')
    parser.add_argument('--codebook_sizes', type=str, default="50,100,200,500",)
    args = parser.parse_args()

    codebook_sizes = [int(x) for x in args.codebook_sizes.split(',') if x.strip()]
    main(model_type=args.model_type, codebook_sizes=codebook_sizes)