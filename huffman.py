import json
import os
import re
import spacy
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
nlp = spacy.load("en_core_web_sm")
import torch
import torch.nn.functional as F
import math

# 自定义字符表（只支持小写字母和数字）
CHARSET = 'abcdefghijklmnopqrstuvwxyz0123456789@._-: /()'
char_to_code = {c: f"{i:06b}" for i, c in enumerate(CHARSET)}
code_to_char = {v: k for k, v in char_to_code.items()}

# chatdocrot
ppl_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
ppl_model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    device_map="auto"
).eval()
# enronqa
# ppl_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
# ppl_model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen1.5-7B-Chat",
#     device_map="auto"
# ).eval()

embed_model = SentenceTransformer('intfloat/e5-base-v2')

import re


def normalize_english_sentence(sentence):
    s = sentence.strip()
    s = re.sub(r"[\u200b\u200c\u200d]", '', s)
    def hyphen_merge(match):

        return re.sub(r"\s*-\s*", "-", match.group(0).replace(" ", ""))

    s = re.sub(r'((?:[A-Za-z]+\s*-\s*)+[A-Za-z]+)', hyphen_merge, s)


    s = re.sub(r"\b([A-Za-z]+)\s+'(s|t|ll|ve|d|re|m)\b", r"\1'\2", s)
    s = re.sub(r"\b([A-Za-z]+)\s+n['’]t\b", r"\1n't", s)
    s = re.sub(r"\b'\s*([A-Za-z]+)\b", r"'\1", s)


    s = re.sub(r"\s+", " ", s)

    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"([.,!?;:])(?=\w)", r"\1 ", s)

    s = s.strip()
    return s




def compute_ppl(text):
    inputs = ppl_tokenizer(text, return_tensors="pt")
    device = next(ppl_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ppl_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

def get_token_distributions(text, return_log=False):
    inputs = ppl_tokenizer(text, return_tensors="pt")
    device = next(ppl_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ppl_model(**inputs)
        logits = outputs.logits[:, :-1, :]  # predict next token
        if return_log:
            return F.log_softmax(logits, dim=-1).squeeze(0)  # [L, V]
        else:
            return F.softmax(logits, dim=-1).squeeze(0)      # [L, V]


def compute_kl(clean_text, stego_text, eps=1e-8, return_avg=False):
    P = get_token_distributions(clean_text, return_log=False)  # [L, V]
    Q = get_token_distributions(stego_text, return_log=False)


    min_len = min(P.size(0), Q.size(0))
    P, Q = P[:min_len, :], Q[:min_len, :]


    P = P.clamp(min=eps)
    Q = Q.clamp(min=eps)


    kl_per_token = (P * (P.log() - Q.log())).sum(dim=1)  # shape: [L]

    if return_avg:
        return kl_per_token.mean().item()
    else:
        return kl_per_token.sum().item()

def compute_js(clean_text, stego_text, eps=1e-8, return_avg=False):
    P = get_token_distributions(clean_text, return_log=False)  # [L, V]
    Q = get_token_distributions(stego_text, return_log=False)

    min_len = min(P.size(0), Q.size(0))
    P, Q = P[:min_len, :], Q[:min_len, :]


    P = P.clamp(min=eps)
    Q = Q.clamp(min=eps)
    M = 0.5 * (P + Q)


    kl_pm = (P * (P.log() - M.log())).sum(dim=1)  # shape: [L]
    kl_qm = (Q * (Q.log() - M.log())).sum(dim=1)  # shape: [L]
    js_per_token = 0.5 * kl_pm + 0.5 * kl_qm      # shape: [L]

    js = js_per_token.mean() if return_avg else js_per_token.sum()


    if torch.isnan(js):
        return 0.0
    else:
        return js.item()


def compute_semantic_similarity(sent1, sent2):
    emb1 = embed_model.encode(sent1, convert_to_tensor=True)
    emb2 = embed_model.encode(sent2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

def better_english_tokenizer(sentence):

    doc = nlp(sentence)
    return [token.text for token in doc]


def text_to_bits_custom(text):

    text = text.lower()
    return ''.join(char_to_code[c] for c in text if c in char_to_code)

def bits_to_text_custom(bits):
    chars = [bits[i:i+6] for i in range(0, len(bits), 6)]
    return ''.join(code_to_char[b] for b in chars if b in code_to_char)

def estimate_embedding_capacity(sentence, codebook, tokenizer=None):
    if tokenizer is None:
        tokenizer = better_english_tokenizer
    tokens = tokenizer(sentence)
    capacity = 0
    for token in tokens:
        token_lower = token.lower()
        if token_lower in codebook:
            code_lengths = [len(code) for code in codebook[token_lower].values()]
            if code_lengths:
                capacity += max(code_lengths)
    return capacity


def hide_bits_in_sentence(sentence, bitstring, codebook, tokenizer=None):
    if tokenizer is None:
        tokenizer = better_english_tokenizer
    tokens = tokenizer(sentence)
    idx = 0
    max_idx = len(bitstring)
    tokens_new = tokens.copy()
    replace_log = []

    for i, token in enumerate(tokens):
        token_lower = token.lower()
        if token_lower in codebook and len(codebook[token_lower]) > 1:
            code2syn = {v: k for k, v in codebook[token_lower].items()}
            for code in sorted(code2syn, key=lambda x: -len(x)):  # 优先使用长码字
                if idx + len(code) <= max_idx and bitstring[idx:idx + len(code)] == code:
                    synonym = code2syn[code]
                    tokens_new[i] = synonym
                    replace_log.append({
                        "origin": token,
                        "new": synonym,
                        "code": code
                    })
                    idx += len(code)
                    break
    new_sentence = ' '.join(tokens_new)
    return new_sentence, idx, replace_log

def extract_bits_from_sentence(_, codebook, replace_log):
    bitstring = ''
    for item in replace_log:
        origin_token = item["origin"].lower()
        replaced_token = item["new"].lower()
        syn2code = codebook.get(origin_token, {})
        for syn, code in syn2code.items():
            if syn.lower() == replaced_token:
                bitstring += code
                break
    return bitstring


def load_codebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        codebook_raw = json.load(f)
    codebook = {}
    for token, syns in codebook_raw.items():
        codebook[token.lower()] = {k.lower(): v for k, v in syns.items()}
    return codebook

def clean_privacys(raw_path, cleaned_path):
    with open(raw_path, 'r', encoding='utf-8') as f:
        secrets_raw = json.load(f)

    secrets = []
    for item in secrets_raw:
        secret = []
        for i in item:
            fields = re.findall(r'"(.*?)"', i) if isinstance(i, str) else []
            if not fields:
                continue
            secret.extend(fields)
        secrets.append(secret)

    with open(cleaned_path, 'w', encoding='utf-8') as f:
        json.dump(secrets, f, indent=2, ensure_ascii=False)
    return secrets

def process_experiment(exp_name, codebook_path):
    m = re.search(r'top(\d+)', os.path.basename(codebook_path))
    if m:
        top_str = m.group(0)
    else:
        top_str = 'top500'

    base_dir = f'./Inputs&Outputs/{exp_name}/Q-R-T-/'
    output_files = glob.glob(os.path.join(base_dir, 'outputs-*.json'))
    if not output_files:
        raise FileNotFoundError(f"No outputs-*.json found in {base_dir}")
    output_path = output_files[0]

    privacy_raw_path = os.path.join(base_dir, 'privacys.json')
    privacy_cleaned_path = os.path.join(base_dir, 'privacys_cleaned.json')
    result_output_path = os.path.join(base_dir, f'embedding_result_{top_str}.json')
    ouput_embed_path = os.path.join(base_dir, f'output_embed_{top_str}.json')
    if not os.path.exists(privacy_cleaned_path):
        clean_privacys(privacy_raw_path, privacy_cleaned_path)

    with open(output_path, 'r', encoding='utf-8') as f:
        outputs = json.load(f)
    with open(privacy_cleaned_path, 'r', encoding='utf-8') as f:
        secrets_all = json.load(f)

    codebook = load_codebook(codebook_path)

    assert len(outputs) == len(secrets_all),

    if os.path.exists(result_output_path):
        with open(result_output_path, 'r', encoding='utf-8') as f:
            result_list = json.load(f)[0:-1]
            for result in result_list:
                if result["delta_ppl"] > 5.0:
                    result["delta_ppl"] = 5.0
    else:
        result_list = []
        output_embed = []
        for i, (sentence, secret_list) in tqdm(enumerate(zip(outputs, secrets_all)), total=len(outputs), desc="Embedding"):
            secret_concat = ' '.join(secret_list).strip().lower()
            bitstring = text_to_bits_custom(secret_concat)
            capacity = estimate_embedding_capacity(sentence, codebook)

            new_sentence, used_bits, replace_log = hide_bits_in_sentence(sentence, bitstring, codebook)
            decoded_bits = extract_bits_from_sentence(new_sentence, codebook, replace_log)
            decoded_secret = bits_to_text_custom(decoded_bits)

            aru = len(decoded_bits) / min(len(bitstring), capacity) if min(len(bitstring), used_bits) > 0 else 0.0

            sentence = normalize_english_sentence(sentence)
            new_sentence = normalize_english_sentence(new_sentence)

            embed_rate = used_bits / len(new_sentence.split()) if len(new_sentence.split()) > 0 else 0.0


            ppl_clean = compute_ppl(sentence)

            ppl_stego = compute_ppl(new_sentence)
            delta_ppl = (ppl_stego - ppl_clean) / ppl_clean if ppl_clean > 0 else 0.0

            if math.isnan(delta_ppl):
                delta_ppl = 0.0

            js_div = compute_js(sentence, new_sentence, return_avg=True)

            sem_sim = compute_semantic_similarity(sentence, new_sentence)

            result_list.append({
                "index": i,
                "original": sentence,
                "embed_capacity": capacity,
                "secret_text": secret_concat,
                "secret_bits": len(bitstring),
                "embedded": new_sentence,
                "bits_used": used_bits,
                "recovered_text": decoded_secret,
                "recovered_bits": len(decoded_bits),
                "recovered_rate": round(len(decoded_bits) / len(bitstring), 4) if len(bitstring) > 0 else 0.0,
                "aru": round(aru, 4),
                "embed_rate": round(embed_rate, 4),
                "delta_ppl": round(delta_ppl, 4),
                "js_div": round(js_div, 4),
                "sem_sim": round(sem_sim, 4)
            })

            output_embed.append(new_sentence)

            with open(result_output_path, 'w', encoding='utf-8') as f:
                json.dump(result_list, f, indent=2, ensure_ascii=False)

            with open(ouput_embed_path, 'w', encoding='utf-8') as f:
                json.dump(output_embed, f, indent=2, ensure_ascii=False)

    ARU_THRESHOLD = 0.5

    total = len(result_list)
    count_above = sum(1 for r in result_list if r['aru'] > ARU_THRESHOLD)

    if total > 0:
        avg_aru = count_above / total
        avg_embed_rate = sum(r['embed_rate'] for r in result_list) / total
        avg_delta_ppl = sum(r['delta_ppl'] for r in result_list) / total
        avg_js_div = sum(r['js_div'] for r in result_list) / total
        avg_sem_sim = sum(r['sem_sim'] for r in result_list) / total

        average_stats = {
            "avg_aru": round(avg_aru, 4),
            "avg_embed_rate": round(avg_embed_rate, 4),
            "avg_delta_ppl": round(avg_delta_ppl, 4),
            "avg_js_div": round(avg_js_div, 4),
            "avg_sem_sim": round(avg_sem_sim, 4)
        }
        result_list.append({"average_stats": average_stats})

        print("Averages:", average_stats)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help="name")
    parser.add_argument('--codebook_path', type=str, default='./Huffman/chatdoctor/huffman_codebook_top50.json')
    args = parser.parse_args()

    process_experiment(args.exp_name, args.codebook_path)