import json
import os
from evalplus.data import get_human_eval_plus

def filter(content):
    i = 0
    j = 0
    max_num = 0
    min_num = 100000
    sub_str = content.split("\n")
    if "def" in content:
        for i in range(len(sub_str)):
            if "def" in sub_str[i]:
                if i < min_num:
                    min_num = i
                elif i > 0 and "```" in sub_str[i-1]:
                    min_num = i
                    break

    if min_num == 100000:
        min_num = 0

    if "return" in content:
        for j in range(len(sub_str)):
            if "return" in sub_str[len(sub_str)-j-1]:
                if len(sub_str)-j-1 > max_num:
                    max_num = len(sub_str)-j-1
                elif j > 0 and "```" in sub_str[len(sub_str)-j]:
                    max_num = len(sub_str)-j-1
                    break
    if max_num == 0:
        max_num = len(sub_str)-1

    return "\n".join(content.split("\n")[min_num+1:max_num+1])

if __name__ == "__main__":
    input_path = "gpt_4_samples.jsonl"
    output_dir = input_path.strip(".jsonl")

    humaneval = {}
    for task_id, problem in get_human_eval_plus().items():
        humaneval[task_id] = problem


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            out_dir = "_".join(data["task_id"].split("/"))
            content = humaneval[data["task_id"]]["prompt"] + filter(data["completion"])

            if not os.path.exists(output_dir+"/"+out_dir):
                os.mkdir(output_dir+"/"+out_dir)
            with open(output_dir+"/"+out_dir+"/0.py", "w") as f1:
                f1.write(content)
                f1.close()
