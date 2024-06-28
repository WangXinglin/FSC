import time

import openai
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm
import json
import sys

class Get:
    def __init__(self):
        self.prompt = ""
    def calc(self, question, gen, temp=0,n=1,model=3.5,fusion_num=5):
            openai.api_type = "azure"  # 'azure', 'azure_ad', 'open_ai'
            openai.api_base = ""
            openai.api_version = "2023-05-15"  # "2023-03-15-preview"
            if model==3.5:
                openai.api_key = ""
                id = "gpt-35-turbo"
            elif model==4:
                openai.api_key =""
                id = "gpt-4"
            else:
                openai.api_key = ""
                id = "gpt-4"

            count = 0
            while True:
                try:
                    content = "I have generated the following responses to the question:\n{}\n{}".format(question, gen)
                    usr = "Please integrate and generate the final response based on the majority consensus of the generated responses."
                    response = openai.ChatCompletion.create(
                        deployment_id=id,
                        messages=[{"role": "system", "content": content},
                                  {"role": "user", "content": usr}],
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
    Gen = Get()
    dataset = ["mbpp", "humaneval", "summscreen", "dm"]
    dataset = dataset[0]
    model = 4
    temp = 0
    n = 1
    samples = []
    i = 0
    fusion_num = 5
    input_path = "dm_gpt_3.5_samples_n_50_temp_0.2.jsonl"

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

    elif dataset == "summscreen":
        input_data = "./data/SummScreen.json"
        with open(input_data, "r") as f:
            data_in = json.load(f)
            f.close()
    elif dataset == "dm":
        input_data = "./data/DM.json"
        with open(input_data, "r") as f:
            data_in = json.load(f)
            f.close()

    with open(input_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if dataset == "mbpp" or dataset == "humaneval":
                task_id_ori = data["task_id"]
                task_id = "/".join(task_id_ori.split("_"))

                question = data_in[task_id]["prompt"]
            elif dataset == "xsum":
                question = "Please summarize the following document:\n" + data["input"]
            elif dataset == "summscreen":
                question = "You are a helpful acodessistant. Please summarize the TV scripts.\n" #+ data["input"]
            elif dataset == "dm":
                question = "Please write a concise summary of following document:\n" + data["input"]

            response = ""
            if dataset == "mbpp" or dataset == "humaneval":
                for i in range(fusion_num):
                    response += "Response {}:\n".format(i+1) + data["completion"][i+random_start] + "\n"

                samples.append(dict(task_id=task_id, completion=Gen.calc(question=question, gen=response, temp=temp, n=n, model=model, fusion_num=fusion_num)[:n]))
            else:
                try:
                    for i in range(fusion_num):
                        response += "Response {}:\n".format(i+1) + data["completion"][i+random_start] + "\n"

                    samples.append(dict(input=data["input"], answer=data["answer"], completion=Gen.calc(question=question, gen=response, temp=temp, n=n, model=model, fusion_num=fusion_num)[:n]))
                except:
                    samples.append(dict(input=data["input"], answer=data["answer"], completion=[" "]))


            write_jsonl("{}_gpt_{}_samples_fusion_{}_start_{}_temp_{}_n_{}.jsonl".format(dataset, model, fusion_num, random_start, temp, n), samples)

    write_jsonl("{}_gpt_{}_samples_fusion_{}_start_{}_temp_{}_n_{}.jsonl".format(dataset, model, fusion_num, random_start, temp, n), samples)
