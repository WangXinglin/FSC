import time

import openai
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm
import json
import re
from copy import deepcopy


def extract_number(text):
    match = re.search('response (\d+)', text)
    match1 = re.search('Response (\d+)', text)
    if match:
        return int(match.group(1))
    elif match1:
        return int(match1.group(1))
    else:
        return None


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
    random_start = choices[3]
    dataset = ["mbpp", "humaneval", "dm", "summscreen"]
    dataset = dataset[1]
    Gen = Get()
    model = 4
    temp = 0
    n = 1
    samples = []
    select_solution = []
    i = 0
    count = 0
    fusion_num = 5
    input_path = "gpt_4_samples_n_100_temp_0.8-sanitized.jsonl"
    if dataset == "mbpp":
        mbpp = {}
        for task_id, problem in get_mbpp_plus().items():
            mbpp[task_id] = problem
        data_in = mbpp
    elif dataset == "humaneval":
        humaneval = {}
        for task_id, problem in get_human_eval_plus().items():
            humaneval[task_id] = problem
        data_in = humaneval


    with open(input_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if dataset == "mbpp" or dataset == "humaneval":
                task_id_ori = data["task_id"]
                task_id = "/".join(task_id_ori.split("_"))

                question = data_in[task_id]["prompt"]
            else:
                question = "Please summarize the following document.\n" + data["input"]
            response = ""
            if dataset == "mbpp" or dataset == "humaneval":
                for i in range(fusion_num):
                    response += "Response {}:\n".format(i+1) + data["completion"][i+random_start]
                samples.append(dict(task_id=task_id,
                                    completion=Gen.calc(question=question, gen=response, temp=temp, n=n, model=model,
                                                        fusion_num=fusion_num)[:n]))
            else:
                try:
                    for i in range(fusion_num):
                        response += "Response {}:\n".format(i+1) + data["completion"][i+random_start] + "\n"

                    samples.append(dict(input=data["input"], answer=data["answer"], completion=Gen.calc(question=question, gen=response, temp=temp, n=n, model=model, fusion_num=fusion_num)[:n]))
                except:
                    samples.append(dict(input=data["input"], answer=data["answer"], completion=[" "]))


            selected = []
            for item in samples[-1]["completion"]:
                choice = extract_number(item.split(".")[0])
                if choice is None:
                    print("N/A")
                    try:
                        selected.append(data["completion"][0+random_start])
                    except:
                        selected.append(" ")
                else:
                    selected.append(data["completion"][choice-1 + random_start])
            if dataset == "mbpp" or dataset == "humaneval":
                select_solution.append(dict(task_id=task_id, completion=deepcopy(selected)))
            else:
                select_solution.append(dict(input=data["input"], answer=data["answer"], completion=deepcopy(selected)))

            write_jsonl("{}_gpt_{}_samples_select_{}_start_{}_temp_{}_n_{}_backup.jsonl".format(dataset, model, fusion_num, random_start, temp, n), samples)
    print(count)
    write_jsonl("{}_gpt_{}_samples_select_{}_start_{}_temp_{}_n_{}.jsonl".format(dataset, model, fusion_num, random_start, temp, n), select_solution)
