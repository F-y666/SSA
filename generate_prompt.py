from retrieval_database import load_retrieval_database_from_parameter, find_all_file, get_encoding_of_file
import os
import json
import re
import random
from nltk.tokenize import RegexpTokenizer
from typing import List, Dict, Any



def get_information():
    def get_target_disease():
        # we can ask ChatGPT to generate the list of different diseases. Then we can get file 'list of disease name.txt'
        with open('Storage/list of disease name.txt', 'r', encoding='utf-8') as file:
            disease = file.read()
        disease = disease.split('\n')
        disease = list(set(disease))
        with open('Information/Target_Disease.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(disease))

    get_target_disease()



def get_question(question_prefix: List[str],
                 question_suffix: List[str],
                 question_adhesive: List[str],
                 question_infor: List[str]) -> Dict[str, List[str]]:
    questions = {}
    _dir = [-1, -1, -1, -1]
    for i, prefix in enumerate(question_prefix):
        if len(question_prefix) != 1:
            _dir[0] = i + 1
        for j, suffix in enumerate(question_suffix):
            if len(question_suffix) != 1:
                _dir[1] = j + 1
            for k, adhesive in enumerate(question_adhesive):
                if len(question_adhesive) != 1:
                    _dir[2] = k + 1
                for l_, infor_name in enumerate(question_infor):
                    if len(question_infor) != 1:
                        _dir[3] = l_ + 1
                    question = []
                    if infor_name.find('Performance') == -1:
                        with open(f'Information/{infor_name}.json') as f_infor:
                            data = json.loads(f_infor.read())
                            random.shuffle(data)

                            data = data[:260]
                            print('****************************************',data[0])
                    else:

                        data_name = infor_name.split('_')[1]
                        with open(f'Data/{data_name}-test/eval_input.json', 'r', encoding='utf-8') as f_infor:
                            data = json.loads(f_infor.read())
                    for infor in data:
                        question.append(prefix + infor + adhesive + suffix)
                    dir_ = [str(s) for s in _dir if s != -1]
                    key = 'Q-' + '+'.join(dir_)
                    questions.update({key: question})
    return questions


def get_contexts(data_name_list: List[List[str]],
                 encoder_model_name: List[str],
                 retrieve_method: List[str],
                 retrieve_num: [int],
                 contexts_adhesive: List[str],
                 threshold: List[float],
                 rerank: List[Any],
                 summarize: List[Any],
                 num_questions: int,
                 questions_dic: Dict[str, List],
                 max_context_length: int = 2048):
    contexts = {}       # used for storage
    contexts_u = {}     # used for generate promote
    sources = {}
    questions = {}

    for key, value in questions_dic.items():
        dir_ = [-1] * 8
        for i1, data_name in enumerate(data_name_list):
            if len(data_name_list) != 1:
                dir_[0] = i1 + 1
            for i2, encoder_model in enumerate(encoder_model_name):
                if len(encoder_model_name) != 1:
                    dir_[1] = i2 + 1
                database = load_retrieval_database_from_parameter(data_name, encoder_model)
                for i3, re_method in enumerate(retrieve_method):
                    if len(retrieve_method) != 1:
                        dir_[2] = i3 + 1
                    for i4, k in enumerate(retrieve_num):
                        if len(retrieve_num) != 1:
                            dir_[3] = i4 + 1
                        # get origin contexts and questions
                        ori_contexts = []
                        all_scores = []
                        ques = []
                        for que in value:
                            ori_context = None
                            now_score = None
                            if re_method == 'mmr':
                                # Note: the mmr method do not have the distance.
                                ori_context = database.max_marginal_relevance_search(que, k=k, fetch_k=10*k)
                            elif re_method == 'knn':
                                ori_context = database.similarity_search_with_score(que, k=k)
                                now_score = [con[1] for con in ori_context]
                                ori_context = [con[0] for con in ori_context]

                            ques.append(que)
                            ori_contexts.append(ori_context)
                            all_scores.append(now_score)
                            if len(ques) == num_questions:
                                break

                        for i5, adhesive in enumerate(contexts_adhesive):
                            if len(contexts_adhesive) != 1:
                                dir_[4] = i5 + 1
                            for i6, now_threshold in enumerate(threshold):
                                if len(threshold) != 1:
                                    dir_[5] = i6 + 1
                                for i7, r_rank in enumerate(rerank):
                                    if len(rerank) != 1:
                                        dir_[6] = i7 + 1
                                    reranker = None
                                    if r_rank == 'yes' or 'bge-reranker-large':
                                        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
                                        # Set use_fp16 to True speed computation with a slight performance degradation
                                    elif r_rank != 'no':
                                        reranker = FlagReranker(f'{r_rank}', use_fp16=True)
                                    for i8, sum_ in enumerate(summarize):
                                        if len(summarize) != 1:
                                            dir_[7] = i8 + 1

                                        _dir = [str(s) for s in dir_ if s != -1]
                                        c_dir = 'R-' + '+'.join(_dir)
                                        cons = []
                                        sour = []
                                        con_u = []
                                        for i, que in enumerate(ques):
                                            ori_context = ori_contexts[i]
                                            # think of threshold to filter the context
                                            if now_threshold != -1 and re_method == 'knn':
                                                now_score = all_scores[i]
                                                ori_context = [con for i, con in enumerate(ori_context) if
                                                               now_score[i] <= now_threshold]
                                            # rerank operation
                                            if r_rank != 'no' and len(ori_context) != 0:
                                                pairs = [(que, con.page_content) for con in ori_context]
                                                scores = reranker.compute_score(pairs)
                                                combined = sorted(zip(ori_context, scores), key=lambda x: x[1])
                                                ori_context = [con for con, score in combined]
                                            t_cons = []
                                            t_sour = []
                                            for con in ori_context:
                                                # we truncate the context to prevent OOM error
                                                if max_context_length != -1:
                                                    t_cons.append(con.page_content[:max_context_length])
                                                else:
                                                    t_cons.append(con.page_content)
                                                t_sour.append(con.metadata['source'])
                                            con_u.append(adhesive.join(t_cons))
                                            cons.append(t_cons)
                                            sour.append(t_sour)
                                        c_dir = key + c_dir
                                        # If summary, because the LLM takes a long time
                                        # the summary will also at the next part
                                        if sum_ != 'no':
                                            con_u = (adhesive, sum_)
                                        contexts.update({c_dir: cons})
                                        contexts_u.update({c_dir: con_u})
                                        sources.update({c_dir: sour})
                                        questions.update({c_dir: ques})
    return contexts, sources, questions, contexts_u


def get_prompt(settings_, output_dir_1) -> List[str]:
    out_lst = []

    print('Step 1: Extract question settings and generate questions')
    ques_set = settings_['question']
    questions = get_question(ques_set['question_prefix'], ques_set['question_suffix'], ques_set['question_adhesive'],
                             ques_set['question_infor'])

    re_set = settings_['retrival']
    contexts, sources, questions, contexts_u = get_contexts(re_set['data_name_list'],
                                                            re_set['encoder_model_name'],
                                                            re_set['retrieve_method'],
                                                            re_set['retrieve_num'],
                                                            re_set['contexts_adhesive'],
                                                            re_set['threshold'],
                                                            re_set['rerank'],
                                                            re_set['summarize'],
                                                            re_set['num_questions'],
                                                            questions,
                                                            re_set['max_context_length'])
    tem_set = settings_['template']
    dir_ = [-1] * 2

    for i1, suf in enumerate(tem_set['suffix']):
        if len(tem_set['suffix']) != 1:
            dir_[0] = i1 + 1
        for i2, adhesive in enumerate(tem_set['template_adhesive']):
            if len(tem_set['template_adhesive']) != 1:
                dir_[1] = i2 + 1
            t_dir = [str(s) for s in dir_ if s != -1]
            p_dir = 'T-' + '+'.join(t_dir)
            for key in contexts:
                context = contexts[key]
                context_u = contexts_u[key]
                source = sources[key]
                question = questions[key]
                n_dir = key + p_dir
                output_dir = f'Inputs&Outputs/{output_dir_1}/{n_dir}'
                prompt = []
                if type(context_u) is not list:
                    # summarize situation
                    prompt = []
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(output_dir + '/set.json', 'w', encoding='utf-8') as file:
                        json.dump({'suffix': suf, 'adhesive_prompt': adhesive, 'adhesive_con': context_u[0],
                                   'infor': context_u[1]}, file)
                else:
                    for i in range(len(question)):
                        prompt.append(suf[0] + context_u[i] + adhesive + suf[1] + question[i] + adhesive + suf[2])
                # store
                out_lst.append(n_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_dir + '/question.json', 'w', encoding='utf-8') as file_q:
                    file_q.write(json.dumps(question))
                with open(output_dir + '/prompts.json', 'w', encoding='utf-8') as file_p:
                    file_p.write(json.dumps(prompt))
                with open(output_dir + '/sources.json', 'w', encoding='utf-8') as file_s:
                    file_s.write(json.dumps(source))
                with open(output_dir + '/context.json', 'w', encoding='utf-8') as file_c:
                    file_c.write(json.dumps(context))

    return out_lst


def get_executable_file(settings_, output_dir_, output_list_, port):
    path = []
    for opt in output_list_:
        path.append(os.path.join(output_dir_, opt))
    # generate bash
    llm_set = settings_['LLM']
    ssa = settings_['SSA']
    with open(f'{output_dir_}.sh', 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n\n')
        for model in llm_set['LLM model']:
            for tem in llm_set['temperature']:
                for top_p in llm_set['top_p']:
                    for max_seq_len in llm_set['max_seq_len']:
                        for max_gen_len in llm_set['max_gen_len']:
                            for opt in path:
                                task = f'python ' \
                                       + f'run_language_model.py --ssa {ssa} ' \
                                       + f'--ckpt_dir {model} --temperature {tem} --top_p {top_p} ' \
                                       + f'--max_seq_len {max_seq_len} --max_gen_len {max_gen_len} --path "{opt}" ;\n'
                                port += 1
                                f.write(task)
    settings_.update({'output_path': path})
    # store the settings
    with open(f'./Inputs&Outputs/{output_dir_}/setting.json', 'w', encoding='utf-8') as file:
        json.dump(settings_, file)


if __name__ == '__main__':
    # Setting parameters
    exp_name = 'chatdoctor-SSA'
    settings = {'question': {'question_prefix': ['I want some advice about '],
                             'question_suffix': ['.'],
                             'question_adhesive': [''],
                             'question_infor': ['Target_Disease']
                             },
                'retrival': {'data_name_list': [['chatdoctor']],
                             'encoder_model_name': ['all-MiniLM-L6-v2'],
                             'retrieve_method': ['knn'],
                             'retrieve_num': [1],
                             'contexts_adhesive': ['\n\n'],
                             'threshold': [-1],
                             'rerank': ['no'],
                             'summarize': ['no'],
                             'num_questions': 250,
                             'max_context_length': 2048
                             },
                'template': {'suffix': [['context: ', 'question: ', 'answer:']],
                             'template_adhesive': ['\n']},
                'LLM': {'LLM model': ['gpt'],
                        'temperature': [0.6],
                        'top_p': [0.9],
                        'max_seq_len': [4096],
                        'max_gen_len': [256]},
                'SSA': True
                }
    master_port = 27000
    # end setting parameters
    # generating the prompts
    print(f'processing {exp_name}')
    output_list = get_prompt(settings, exp_name)
    get_executable_file(settings, exp_name, output_list, master_port)
