import os
from tqdm import tqdm
import random
import shutil
import json
import argparse
from typing import List
from chardet.universaldetector import UniversalDetector
import pandas as pd
import torch
import langchain
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from nltk.tokenize import RegexpTokenizer


def find_all_file(path: str) -> List[str]:
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def get_encoding_of_file(path: str) -> str:
    detector = UniversalDetector()
    with open(path, 'rb') as file:
        data = file.readlines()
        for line in data:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']


def get_embed_model(encoder_model_name: str,
                    device: str = 'cpu',
                    retrival_database_batch_size: int = 256) -> OpenAIEmbeddings:
    if encoder_model_name == 'open-ai':
        embed_model = OpenAIEmbeddings()
    else:
        if os.path.exists(encoder_model_name) and os.path.isdir(encoder_model_name):
            model_name = encoder_model_name
        elif encoder_model_name == 'all-MiniLM-L6-v2':
            model_name = 'all-MiniLM-L6-v2'
        elif encoder_model_name == 'bge-large-en-v1.5':
            model_name = 'BAAI/bge-large-en-v1.5'
        elif encoder_model_name == 'e5-base-v2':
            model_name = 'intfloat/e5-base-v2'
        else:
            model_name = encoder_model_name

        embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size}
        )

    return embed_model


def pre_process_dataset(data_name: str, change_method: str = 'body') -> None:
    data_store_path = 'Data'

    def pre_process_chatdoctor() -> None:
        file_path = os.path.join(data_store_path, 'chatdoctor200k/chatdoctor200k.json')
        with open(file_path, 'r') as f:
            content = f.read()
            data = json.loads(content)
        output_path = os.path.join(data_store_path, 'chatdoctor/chatdoctor.txt')
        with open(output_path, 'w', encoding="utf-8") as f:
            max_len = 0
            for i, item in enumerate(data):
                s = 'input: ' + item['input'] + '\n' + 'output: ' + item['output']
                s = s.replace('\xa0', ' ')
                if i != len(data) - 1:
                    s += '\n\n'
                max_len = max(max_len, len(s))
                f.write(s)
        print(f'Number of chatdoctor dataset is {len(data)}')  # 207408
        print(f'Max length of chatdoctor dataset is {max_len}')  # 11772


    def pre_process_enron_qa() -> None:
        file_path = os.path.join(data_store_path, 'enron-qa/')
        dev = pd.read_parquet(os.path.join(file_path, 'dev-00000-of-00001.parquet'))
        test = pd.read_parquet(os.path.join(file_path, 'test-00000-of-00001.parquet'))
        train1 = pd.read_parquet(os.path.join(file_path, 'train-00000-of-00002.parquet'))
        train2 = pd.read_parquet(os.path.join(file_path, 'train-00001-of-00002.parquet'))
        train = pd.concat([train1, train2], ignore_index=True)
        df = pd.concat([train, dev, test], ignore_index=True)

        output_email_dir = os.path.join(data_store_path, 'enron-qa-email')
        os.makedirs(output_email_dir, exist_ok=True)

        output_question_path = os.path.join(data_store_path, 'enron-qa/all_questions.json')
        questions = []

        for idx, row in df.iterrows():
            email = str(row['email']).strip()
            question_data = row['questions']
            if hasattr(question_data, '__getitem__') and len(question_data) > 0:
                question = str(question_data[0]).strip()
            else:
                question = ''
            with open(os.path.join(output_email_dir, f"{idx}.txt"), 'w', encoding='utf-8') as f:
                f.write(email)
            questions.append(question)

        with open(output_question_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

    if data_name == "chatdoctor200k":
        pre_process_chatdoctor()
    elif data_name == "enron-qa":
        pre_process_enron_qa()


def split_dataset(data_name: str, split_ratio: int = 0.99, num_eval: int = 1000, max_que_len: int = 50) -> None:
    data_store_path = 'Data'
    if data_name == 'chatdoctor':
        with open('Data/chatdoctor/chatdoctor.txt', 'r', encoding="utf-8") as f:
            data = f.read()
        data = data.split('\n\n')
        output_train_path = os.path.join(data_store_path, 'chatdoctor-train/chatdoctor.txt')
        output_test_path = os.path.join(data_store_path, 'chatdoctor-test/chatdoctor.txt')
        num_ = int(split_ratio * len(data))
        random.shuffle(data)
        with open(output_train_path, 'w', encoding="utf-8") as f:
            f.write('\n\n'.join(data[:num_]))
        with open(output_test_path, 'w', encoding="utf-8") as f:
            f.write('\n\n'.join(data[num_:]))
        # getting information of performance evaluation
        test_data = data[num_:]
        random.shuffle(test_data)
        eval_data = test_data[:num_eval]
        eval_input = []
        eval_output = []
        for e_data in eval_data:
            item = e_data.split('\noutput: ')
            eval_input.append(item[0][7:])
            eval_output.append(item[1])
        with open(f'Data/{data_name}-test/eval_input.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_input))
        with open(f'Data/{data_name}-test/eval_output.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_output))
    else:
        """
        If a dataset is stored in multiple files, and you only want to partition the dataset at the file level
        You can use this code directly
        Alternatively, you can modify this section of the code according to your specific requirements.
        """
        data_path = os.path.join(data_store_path, data_name)
        all_file = []
        for file_name in find_all_file(data_path):
            all_file.append(file_name)
        random.shuffle(all_file)
        num_train = int(len(all_file) * split_ratio)
        train_all_file = all_file[:num_train]
        test_all_file = all_file[num_train:]
        print(f'Number of the training set is {len(train_all_file)}, number of the test set is {len(test_all_file)}')
        for train_file in train_all_file:
            source_file = train_file    # source path
            target_file = train_file.replace(data_name, f'{data_name}-train')
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            # using shutil to copy file
            shutil.copy2(source_file, target_file)

        for test_file in test_all_file:
            source_file = test_file    # source path
            target_file = test_file.replace(data_name, f'{data_name}-test')
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            shutil.copy2(source_file, target_file)
        # generating input for performance evaluation
        random.shuffle(test_all_file)
        eval_data = test_all_file[:num_eval]
        eval_input = []
        tokenizer = RegexpTokenizer(r'\w+')
        for path_eval_data in eval_data:
            encoding = get_encoding_of_file(path_eval_data)
            with open(path_eval_data, 'r', encoding=encoding) as file:
                data = file.read()
            que = tokenizer.tokenize(data)[:max_que_len]
            eval_input.append(' '.join(que))
        with open(f'Data/{data_name}-test/eval_input.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_input))


def construct_retrieval_database(data_name_list: List[str],
                                 split_method: List[str] = None,
                                 encoder_model_name: str = 'all-MiniLM-L6-v2',
                                 retrival_database_batch_size: int = 256,
                                 chunk_size: int = 1500,
                                 chunk_overlap: int = 100,
                                 ) -> 'langchain.vectorstores.chroma.Chroma':
    class SingleFileSplitter(TextSplitter):
        def split_text(self, text: str) -> List[str]:
            return [text]

    class LineBreakTextSplitter(TextSplitter):
        def split_text(self, text: str) -> List[str]:
            return text.split("\n\n")

    def get_splitter(split_method_) -> SingleFileSplitter:
        splitter_ = None
        if split_method_ == 'single_file':
            splitter_ = SingleFileSplitter()
        elif split_method_ == 'by_two_line_breaks':
            splitter_ = LineBreakTextSplitter()
        elif split_method_ == 'recursive_character':
            splitter_ = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter_

    data_store_path = 'Data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if split_method is None:
        # No split method provided, default method used
        split_method = ['single_file'] * len(data_name_list)
    elif len(split_method) == 1:
        # Only one split method is provided, this method is used for all the datasets
        split_method = split_method * len(data_name_list)
    else:
        assert len(split_method) == len(data_name_list)
    split_texts = []
    for n_data_name, data_name in enumerate(data_name_list):
        documents = []
        # open the files
        data_path = os.path.join(data_store_path, data_name)
        for file_name in tqdm(list(find_all_file(data_path)), desc=f"Loading files from {data_path}"):
            # detect the encode method of files:
            encoding = get_encoding_of_file(file_name)
            # load the data
            loader = TextLoader(file_name, encoding=encoding)
            doc = loader.load()
            documents.extend(doc)

        print(f'File number of {data_name}: {len(documents)}')
        # get the splitter
        splitter = get_splitter(split_method[n_data_name])
        # split the texts
        split_texts += splitter.split_documents(documents)
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    retrieval_name = '_'.join(data_name_list)
    if len(data_name_list) != 1:
        retrieval_name = 'mix_' + retrieval_name
    vector_store_path = f"./RetrievalBase/{retrieval_name}/{encoder_model_name}"
    print(f'generating chroma database of {retrieval_name} using {encoder_model_name}')
    retrieval_database = Chroma.from_documents(documents=split_texts,
                                               embedding=embed_model,
                                               persist_directory=vector_store_path)
    print("**********************Done**********************************************")
    return retrieval_database


def load_retrieval_database_from_address(store_path: str,
                                         encoder_model_name: str = 'all-MiniLM-L6-v2',
                                         retrival_database_batch_size: int = 512
                                         ) -> 'langchain.vectorstores.chroma.Chroma':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=store_path
    )
    return retrieval_database


def load_retrieval_database_from_parameter(data_name_list: List[str],
                                           encoder_model_name: str = 'all-MiniLM-L6-v2',
                                           retrival_database_batch_size: int = 512
                                           ) -> 'langchain.vectorstores.chroma.Chroma':
    database_store_path = 'RetrievalBase'
    retrieval_name = '_'.join(data_name_list)
    if len(data_name_list) != 1:
        retrieval_name = 'mix_' + retrieval_name
    store_path = f"./{database_store_path}/{retrieval_name}/{encoder_model_name}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=store_path
    )
    return retrieval_database


if __name__ == '__main__':
    """
    You can run following example code to do your experiments, or directly parse parameters
    
    # preprocessing the data
    # pre_process_dataset('chatdoctor200k')           
    # run this code to edit the enronqa data
    # pre_process_dataset('enron-qa')

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--encoder_model', type=str)
    parser.add_argument('--flag_mix', type=bool, default=False)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    encoder_model = args.encoder_model
    flag_mix = args.flag_mix

    if dataset_name.find('enron-qa-email') != -1:
        construct_retrieval_database([dataset_name], ['single_file'], encoder_model)
    elif dataset_name.find('chatdoctor') != -1:
        construct_retrieval_database([dataset_name], ['by_two_line_breaks'], encoder_model)
