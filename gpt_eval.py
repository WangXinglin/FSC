import json
import time

import openai
from tqdm import tqdm
import os
import argparse
import gzip
from typing import Dict, Iterable

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))

class Get:
    def __init__(self):
        self.prompt = ""
    def calc(self,doc="", input="", temp=0,n=1,model=3.5, task="code"):
            openai.api_type = "azure"  # 'azure', 'azure_ad', 'open_ai'
            openai.api_base = "https://runway.devops.xiaohongshu.com"
            openai.api_version = "2023-05-15"  # "2023-03-15-preview"
            if model==3.5:
                openai.api_key = "please input your token here"
                id = "gpt-35-turbo"
            elif model==4:
                openai.api_key = "please input your token here"
                id = "gpt4-PTU"
            else:
                openai.api_key = "please input your token here"
                id = "gpt-4"


            if task == "summ_screen":
                instruction = "You will be given one summary written for a TV show episode.\nYour task is to rate the summary on one metric.\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nEvaluation Criteria:\nCoherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby \"the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.\"\n\nEvaluation Steps:\n1. Read the news article carefully and identify the main topic and key points.\n2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\nExample:\nSource Text:\n{" + doc + "}" + "\nSummary:\n{" + input + "}\nEvaluation Form (scores ONLY):\n- Coherence:"
                cur = []
            elif task == "DM":
                instruction = "You will be given one summary written for a TV show episode.\nYour task is to rate the summary on one metric.\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nEvaluation Criteria:\nCoherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby \"the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.\"\n\nEvaluation Steps:\n1. Read the news article carefully and identify the main topic and key points.\n2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\nExample:\nSource Text:\n{" + doc + "}" + "\nSummary:\n{" + input + "}\nEvaluation Form (scores ONLY):\n- Coherence:"
                cur = []

            count = 0
            while True:
                messages = [{"role": "system", "content": instruction}]

                try:
                    response = openai.ChatCompletion.create(
                        deployment_id=id,
                        messages=messages,
                        top_p=1,
                        n=n,
                        temperature=temp,
                    )
                    res = [tem["message"]["content"] for tem in response["choices"]]
                    return res
                except Exception as e:
                    if count > 2:
                        return [" "]
                    print("An error occurred:", e)
                    count += 1
                    time.sleep(18)

def str_to_float(input_list):
    out = []
    for item in input_list:
        try:
            out.append(float(item))
        except:
            m = 1

    return sum(out)/len(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help="input jsonl path", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    input_data = []
    with open(input_path, "r") as f:
        for line in tqdm(f):
            sub_data = json.loads(line)
            input_data.append(sub_data)

    Gen = Get()
    model = 4
    temp = 1
    n = 20
    samples = []
    i = 0
    dataset = ["summ_screen", "DM"]

    dataset = dataset[0]

    output_path = input_path.split(".json")[0] + "gpt{}eval".format(model) + ".jsonl"
    for item in tqdm(input_data):
        scores = []
        for i in range(len(item["completion"][:1])):
            completion = Gen.calc(doc=item["input"], input=item["completion"][i],temp=temp, n=n, model=model, task=dataset)[:n]
            try:
                scores.append(str_to_float(completion))
            except:
                print("bad")
        try:
            samples.append(dict(input=item["input"], answer=item["answer"], score=sum(scores)/len(scores)))
        except:
            print("N/A")
        write_jsonl(output_path, samples)
    write_jsonl(output_path, samples)



