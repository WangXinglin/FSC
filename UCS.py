import time

import openai
from evalplus.data import get_human_eval_plus, write_jsonl
from tqdm import tqdm
import json
import re
from copy import deepcopy
from collections import Counter


def extract_number(text):
    match = re.search('response (\d+)', text)
    match1 = re.search('Response (\d+)', text)
    if match:
        return int(match.group(1))
    elif match1:
        return int(match1.group(1))
    else:
        return None

def compute_1gram_sum(S):
    # Split each string into words and count the occurrence of each word
    word_counts = Counter(word for s in S for word in s.split())

    # Compute the 1-gram sum for each string
    result = {}
    for s in S:
        if len(s.split()) > 0:
            result[s] = sum(word_counts[word] for word in s.split()) / len(s.split())

    return result

def cal_USC(input_list):
    result = compute_1gram_sum(input_list)

    sorted_input_list = sorted(result.items(), key=lambda x: x[1], reverse=True)

    sorted_keys = [item[0] for item in sorted_input_list]

    return sorted_keys


class Get:
    def __init__(self):
        self.prompt = ""
    def calc(self,question, gen, temp=0,n=1,model=3.5,fusion_num=5):
            openai.api_type = ""  # 'azure', 'azure_ad', 'open_ai'
            openai.api_base = ""
            openai.api_version = "2023-05-15"  # "2023-03-15-preview"
            if model==3.5:
                openai.api_key = ""
                id = "gpt-35-turbo"
            elif model==4:
                openai.api_key = ""
                id = "gpt-4"
            else:
                openai.api_key = ""
                id = "gpt-4"

            count = 0
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        deployment_id=id,
                        messages=[{"role": "system", "content": "I have generated the following responses to the question {}.\n {}".format(question, gen)},
                                  {"role": "user", "content": "Select the most consistent response based on majority consensus. Start your answer with \"The most consistent response is response x.\""}],
                        top_p=0.9,
                        n=n,
                        temperature=temp,
                    )
                    res = [tem["message"]["content"] for tem in response["choices"]]
                    return res
                except Exception as e:
                    if count > 5:
                        return [" "]
                    print("An error occurred:", e)
                    count += 1
                    time.sleep(18)


if __name__ == "__main__":
    choices = [0, 30, 60, 90]
    random_start = choices[1]
    dataset = ["mbpp", "humaneval", "dm", "summscreen"]
    dataset = dataset[3]
    Gen = Get()
    model = 3.5
    temp = 0
    n = 1
    samples = []
    select_solution = []
    i = 0
    count = 0
    fusion_num = 5
    input_path = "summ_gpt_3.5_samples_n_50_temp_0.8.jsonl"


    with open(input_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if dataset == "mbpp" or dataset=="humaneval":
                task_id_ori = data["task_id"]
                task_id = "/".join(task_id_ori.split("_"))

            result = cal_USC(data["completion"][random_start: random_start+fusion_num])
            try:
                selected = [result[0]]
            except:
                print("N/A")
                selected = [""]

            if dataset == "mbpp" or dataset=="humaneval":
                select_solution.append(dict(task_id=task_id, completion=deepcopy(selected)))
            else:
                select_solution.append(dict(input=data["input"], answer=data["answer"], completion=deepcopy(selected)))

    write_jsonl("{}_gpt_{}_samples_sc_{}_start_{}_temp_{}_n_{}.jsonl".format(dataset, model, fusion_num, random_start, temp, n), select_solution)
