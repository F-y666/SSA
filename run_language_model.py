import fire
from llama import Llama
import warnings
import json
import os
from langchain_openai import OpenAI
import transformers
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError, Timeout
from langchain_openai import ChatOpenAI as LangOpenAI

import time
from pathlib import Path
"""
This file is to run large language model.
The running instructions have been generated in file f'{experiment name}.sh'
Please run the following command:
nohup bash {experiment name}.sh > output_name.out
or: bash {experiment name}.sh
"""
# If you want to use OpenAI's model, please set API here

def safe_invoke(llm, prompt, retries=3, delay=2):
    for i in range(retries):
        try:
            return llm.invoke(prompt).content
        except (APIConnectionError, RateLimitError, Timeout) as e:
            print(f"[重试 {i+1}/{retries}] OpenAI 请求失败：{type(e).__name__}: {e}")
            time.sleep(delay)
    print("[跳过] 超过最大重试次数。已返回 [NO_PRIVACY]")
    return "[NO_PRIVACY]"


def main(
        ckpt_dir: str,       # LLM model name
        path: str,           # input and output place
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 4096,
        max_gen_len: int = 256,
        ssa: bool = False,
):
    print(path)
    generator = None
    llm = None
    # summary stage
    if os.path.exists(f'./Inputs&Outputs/{path}/set.json'):
        print('summarizing now')
        # need to summarize
        with open(f'./Inputs&Outputs/{path}/set.json', "r") as file:
            settings = json.load(file)
        summary_model = settings['infor']
        print(summary_model)

        para_flag = False
        if summary_model.find('-para'):
            para_flag = True
            summary_model = summary_model.strip('-para')

        tokenizer = AutoTokenizer.from_pretrained('Model/' + summary_model, use_fast=False)
        pipe = transformers.pipeline(
            "text-generation",
            model='Model/' + summary_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

        suf = settings['suffix']
        adh_1 = settings['adhesive_con']
        adh_2 = settings['adhesive_prompt']
        with open(f"./Inputs&Outputs/{path}/question.json", 'r', encoding='utf-8') as f_que:
            questions = json.loads(f_que.read())
        with open(f"./Inputs&Outputs/{path}/context.json", 'r', encoding='utf-8') as f_con:
            contexts = json.loads(f_con.read())

        su_1 = "Given the following question and context, extract any part of the context" \
               + " *AS IS* that is relevant to answer the question. If none of the context is relevant" \
               + " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Question: "
        if para_flag:
            su_1 = "Given the following question and context, extract any part of the context" \
                   + " *AS IS* that is relevant to answer the question. If none of the context is relevant" \
                   + " return NO_OUTPUT.\n\n> Question: "
        su_2 = "\n> CONTEXT: {"
        su_3 = " }\n>>>\nExtracted relevant parts:"

        summary_path = f"./Inputs&Outputs/{path}/summarize_contexts.json"
        prompt_path = f"./Inputs&Outputs/{path}/generate_summarize_prompt.json"

        if Path(summary_path).exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summarize_contexts = json.load(f)
        else:
            summarize_contexts = []

        if Path(prompt_path).exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_ge_contexts = json.load(f)
        else:
            prompt_ge_contexts = []


        while len(summarize_contexts) < len(questions):
            summarize_contexts.append([])
            prompt_ge_contexts.append([])


        for i in tqdm(range(len(questions)), desc="Summarizing"):
            if summarize_contexts[i]:
                continue
            ques = questions[i]
            k_contexts = contexts[i]
            ge_contexts = []
            sum_contexts = []

            for j in range(len(k_contexts)):
                context = k_contexts[j]
                prompt_ge_context = su_1 + ques + su_2 + context + su_3
                ge_contexts.append(prompt_ge_context)
                common_kwargs = {
                    "do_sample": True,
                    "max_new_tokens": 256,
                    "temperature": 0.3,
                    "top_p": top_p,
                    "num_return_sequences": 1,
                    "return_full_text": False,
                }
                results = pipe(prompt_ge_context, **common_kwargs, pad_token_id=tokenizer.eos_token_id)
                ans = results[0]['generated_text']
                sum_contexts.append(ans)

            summarize_contexts[i] = sum_contexts
            prompt_ge_contexts[i] = ge_contexts

            with open(summary_path, 'w', encoding='utf-8') as f_c:
                json.dump(summarize_contexts, f_c, indent=2)
            with open(prompt_path, 'w', encoding='utf-8') as f_g:
                json.dump(prompt_ge_contexts, f_g, indent=2)


        prompts = []
        for i in range(len(questions)):
            con_u = adh_1.join(summarize_contexts[i])
            prompt = suf[0] + con_u + adh_2 + suf[1] + questions[i] + adh_2 + suf[2]
            prompts.append(prompt)
        with open(f"./Inputs&Outputs/{path}/prompts.json", 'w', encoding='utf-8') as f_p:
            f_p.write(json.dumps(prompts))

    if ssa:
        print('extracting now')
        client = OpenAI()

        with open(f"./Inputs&Outputs/{path}/context.json", 'r', encoding='utf-8') as f_con:
            contexts = json.loads(f_con.read())

        su_6 = "\n> CONTEXT: {"
        su_4 = """
        Below is a medical dialogue between a patient and a doctor. Extract all personal privacy information **only from the patient's statements**. Ignore the doctor's replies.

Output a single JSON array containing only the pieces of personal privacy (e.g., diseases, symptoms, medications, lab values, behaviors, emotions, lifestyle, etc.) mentioned by the patient. Do not include any text from the doctor. Do not return explanations or any additional formatting.

If there is no personal privacy information, return: [NO_PRIVACY]

        """
#         su_4 = (
#             "You are given the full content of an internal email thread. "
#             "Extract all names of people and email addresses mentioned anywhere in the message.\n\n"
#             "Include:\n"
#             "- Full names (e.g., 'Kevin Presto', 'Panus, Stephanie')\n"
#             "- Partial names (e.g., 'David', 'Portz') if they refer to people\n"
#             "- Email addresses (e.g., 'stephanie.panus@enron.com')\n\n"
#             "Ignore titles, roles, companies, or unrelated words.\n\n"
#             "Output a JSON array of strings. If none, return [NO_PRIVACY]. "
#             "Do not include any explanation or formatting outside the array."
#         )

        su_5 = " }\n Only output the JSON array and nothing else."

        privacy_path = f"./Inputs&Outputs/{path}/privacys.json"
        prompt_path = f"./Inputs&Outputs/{path}/privacy_ex_prompt.json"

        if Path(privacy_path).exists():
            with open(privacy_path, 'r', encoding='utf-8') as f:
                privacys = json.load(f)
        else:
            privacys = []

        if Path(prompt_path).exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                privacy_ex_prompts = json.load(f)
        else:
            privacy_ex_prompts = []

        while len(privacys) < len(contexts):
            privacys.append([])
            privacy_ex_prompts.append([])

        for i in tqdm(range(len(contexts)), desc="SSAing"):
            if privacys[i]:
                continue

            k_contexts = contexts[i]
            ex_prompts = []
            privacy = []
            for context in k_contexts:
                prompt = su_4 + su_6 + context + su_5
                ex_prompts.append(prompt)
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo", # o1-mini
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_gen_len
                )
                pri = completion.choices[0].message.content
                privacy.append(pri)

            privacys[i] = privacy
            privacy_ex_prompts[i] = ex_prompts

            with open(privacy_path, 'w', encoding='utf-8') as f_c:
                json.dump(privacys, f_c, indent=2)
            with open(prompt_path, 'w', encoding='utf-8') as f_g:
                json.dump(privacy_ex_prompts, f_g, indent=2)

    flag_llm = 'llama'
    if ckpt_dir.find('gpt') != -1:
        if ckpt_dir == 'gpt':
            ckpt_dir = 'gpt-3.5-turbo'
        client = OpenAI()
        flag_llm = 'gpt'
    else:
        tokenizer = AutoTokenizer.from_pretrained('Model/' + ckpt_dir, use_fast=False)
        pipe = transformers.pipeline(
            "text-generation",
            model='Model/' + ckpt_dir,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
    # generate output
    print('generating output')
    with open(f"./Inputs&Outputs/{path}/prompts.json", 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())

    output_path = f"./Inputs&Outputs/{path}/outputs-{ckpt_dir}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"

    if Path(output_path).exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            answer = json.load(f)
    else:
        answer = []

    while len(answer) < len(all_prompts):
        answer.append("")

    for i in tqdm(range(len(all_prompts)), desc="Generating output"):
        if answer[i]:  # 已完成
            continue
        prompt = all_prompts[i]
        if flag_llm == 'gpt':
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_gen_len
            )
            ans = completion.choices[0].message.content
        else:
            common_kwargs = {
                "do_sample": True,
                "max_new_tokens": max_gen_len,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": 1,
                "return_full_text": False,
            }
            results = pipe(prompt, **common_kwargs, pad_token_id=tokenizer.eos_token_id)
            ans = results[0]['generated_text']
        answer[i] = ans

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(answer, file, indent=2)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
