 # Silent Substitutions: Exploiting Auxiliary Modules for Stealthy Privacy Attacks in Retrieval-Augmented Generation


## Abstract

Retrieval-Augmented Generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external documents at inference time to produce more accurate and contextually grounded responses. Despite the benefits, this retrieval process may inadvertently expose sensitive information, posing significant privacy risks. Specifically, prior attacks attempt to elicit verbatim reproduction of the retrieved context, risking the disclosure of private information. To counter this, lexical auditing techniques have been proposed, but their effectiveness remains unclear. In this work, we revisit the above problem and identify a previously overlooked threat: an insider adversary can steal private information by manipulating auxiliary components of the RAG pipeline. To demonstrate this, we propose Synonym-based Stego Attack (SSA), a modular and stealthy method that encodes private content into model outputs using synonym substitutions guided by Huffman coding, thereby establishing a covert leakage channel. SSA requires no changes to the core RAG architecture or adversarial queries, relying solely on auxiliary modules to carry out privacy extraction and embedding. We evaluate SSA on two real-world RAG applications. It achieves over 59\% leakage success under ROUGE-L-based defenses—far surpassing baselines ($<$15\%)—while preserving linguistic quality with minimal changes in perplexity ($\Delta$PPL $<$ 0.80) and stylistic divergence (JS $<$ 0.60), making it both effective and stealthy.


## About the  model and data
```
|-- Model
    |-- llama-3-8B
```

```
|-- Data
    |-- chatdoctor
    |-- enron-qa-mail
```


## Examples and illustrate

There are 4 steps to run the experiment: retrieval database, generate prompt, run language model, and evaluation results. 

### 1. retrieval database

In this section, we perform pre-processing on the datasets and construct the vector database.

You can use the following code to construct the database for the training set of `chatdoctor` using the `bge-large-en-v1.5` model.

```
python retrieval_database.py \
--dataset_name="chatdoctor-train" \
--encoder_model="bge-large-en-v1.5"
```

### 2. generate prompt

To run attack, you can run following codes.

```
python generate_prompt.py
```

### 3. run language model

```
sh ./{exp_name}.sh
```

### 4. evaluation results

After the previous part of the code has finished running, you can use the following code to evaluate the results:

```
python eval.py 
```



