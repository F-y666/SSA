import random
import fasttext
import json
import glob
import os
from sklearn.metrics import accuracy_score, f1_score

root_dir = './Inputs&Outputs'
pattern = os.path.join(root_dir, 'ablation-enronqa-SSA-*', 'Q-R-T-')
stego_files = glob.glob(os.path.join(pattern, 'output_embed_top100.json'))
normal_files = glob.glob(os.path.join(pattern, 'outputs-Llama-3.1-8B-Instruct-*.json'))

def extract_texts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []

train_lines = []

normal_texts = []
for f in normal_files:
    for text in extract_texts(f):
        text = text.strip().replace('\n', ' ')
        normal_texts.append(f'__label__0 {text}')

stego_texts = []
for f in stego_files:
    for text in extract_texts(f):
        text = text.strip().replace('\n', ' ')
        stego_texts.append(f'__label__1 {text}')

min_count = int (min(len(normal_texts), len(stego_texts)) * 0.03)

random.seed(42)
sampled_normal = random.sample(normal_texts, min_count)
sampled_stego = random.sample(stego_texts, min_count)

train_lines = sampled_normal + sampled_stego
random.shuffle(train_lines)

with open('train.txt', 'w', encoding='utf-8') as f:
    for line in train_lines:
        f.write(line + '\n')

model = fasttext.train_supervised(input="train.txt", epoch=3, lr=0.1, wordNgrams=1, verbose=2)

# 保存模型
model.save_model("fasttext-binary-model.bin")


model = fasttext.load_model('fasttext-binary-model.bin')


test_dir = './Inputs&Outputs/enronqa-SSA/Q-R-T-'


test_files = []
test_files += glob.glob(os.path.join(test_dir, 'output_embed_top50.json'))
test_files += glob.glob(os.path.join(test_dir, 'outputs-Llama-3.1-8B-Instruct-*.json'))

for f in test_files:
    print('  ', f)

all_texts = []
true_labels = []
for f in test_files:
    with open(f, 'r', encoding='utf-8') as jf:
        texts = json.load(jf)
        all_texts.extend(texts)
        if os.path.basename(f).startswith('output_embed'):
            true_labels.extend(['__label__1'] * len(texts))
        else:
            true_labels.extend(['__label__0'] * len(texts))

all_texts = [t.replace('\n', ' ') for t in all_texts]
pred_labels, probs = model.predict(all_texts)
pred_labels = [l[0] for l in pred_labels]

acc = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, pos_label='__label__1')

