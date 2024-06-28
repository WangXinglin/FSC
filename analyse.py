import os.path
import time

from tqdm import tqdm
import json
import sys
from rouge import Rouge
from rouge_score import rouge_scorer
from bert_score import score

def calculate_bertscore(s, t):
    P, R, F1 = score([s], [t], lang="en", verbose=True, model_type="roberta-large")
    return P, R, F1
def cal_metrics(input, answer_list):
    sub_metric = {"rouge-1":{"r":[], "p":[], "f":[]}, "rouge-2":{"r":[], "p":[], "f":[]}, "rouge-l": {"r":[], "p":[], "f":[]}, "bert-score":{"r":[], "p":[], "f":[]}}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    for answer in answer_list:
        #scores = rouge.get_scores(item, answer)
        scores = scorer.score(answer, input)
        sub_metric["rouge-1"]["r"].append(scores['rouge1'].recall)
        sub_metric["rouge-1"]["p"].append(scores['rouge1'].precision)
        sub_metric["rouge-1"]["f"].append(scores['rouge1'].fmeasure)

        sub_metric["rouge-2"]["r"].append(scores['rouge2'].recall)
        sub_metric["rouge-2"]["p"].append(scores['rouge2'].precision)
        sub_metric["rouge-2"]["f"].append(scores['rouge2'].fmeasure)

        sub_metric["rouge-l"]["r"].append(scores['rougeL'].recall)
        sub_metric["rouge-l"]["p"].append(scores['rougeL'].precision)
        sub_metric["rouge-l"]["f"].append(scores['rougeL'].fmeasure)

        bert_p, bert_r, bert_f = 0,0,0 #calculate_bertscore(item, answer)
        sub_metric["bert-score"]["r"].append(bert_r)
        sub_metric["bert-score"]["p"].append(bert_p)
        sub_metric["bert-score"]["f"].append(bert_f)

    for key in sub_metric.keys():
        try:
            for sub_key in sub_metric[key].keys():
                sub_metric[key][sub_key].remove(max(sub_metric[key][sub_key]))
                sub_metric[key][sub_key] = sum(sub_metric[key][sub_key]) / len(sub_metric[key][sub_key])
        except:
            sub_metric[key] = sum(sub_metric[key]) / len(sub_metric[key])

    return sub_metric



if __name__ == "__main__":

    method_path = "humaneval_gpt_3.5_samples_fusion_5_start_90_temp_0_n_1_new-sanitized.jsonl" #"xsum_gpt_4_samples_select_5_start_30_temp_0_n_1.jsonl" #"xsum_gpt_3.5_samples_fusion_5_start_30_temp_0_n_1_wo.jsonl" #"xsum_gpt_4_samples_fusion_5_start_30_temp_0_n_1.jsonl" #"xsum_gpt_3.5_samples_fusion_5_start_30_temp_0_n_1_commonality.jsonl" #"xsum_gpt_3.5_samples_select_5_start_30_temp_0_n_1.jsonl"#"xsum_gpt_3.5_samples_fusion_5_start_30_temp_0_n_1.jsonl" #"xsum_gpt_3.5_samples_n_50_temp_0.2.jsonl" #"xsum_gpt_3.5_samples_fusion_5_start_30_temp_0_n_1_new.jsonl" # xsum_gpt_3.5_samples_n_50_temp_0.2_new.jsonl"#"xsum_gpt_3.5_samples_fusion_5_start_30_temp_0_n_1_new.jsonl" #"xsum_gpt_3.5_samples_n_50_temp_0.2_new.jsonl"  # "mbpp_gpt_3.5_samples_n_100_temp_0.8_new-sanitized.jsonl"
    baseline_path = "gpt_3.5_samples_n_100_temp_0.8-sanitized.jsonl"
    start = 90
    fusion_num = 5

    output_path = "analysis_neighbour_result/"+method_path.split(".jsonl")[0] + ".json"
    # out_dataset = "./humaneval.json"
    # with open(out_dataset, "w") as f:
    #     json.dump(data_in, f)
    #     f.close()
    #
    # sys.exit()
    baseline = {}
    with open(baseline_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            baseline[data["task_id"]] = data["completion"][start: start+fusion_num]
        f.close()


    if os.path.exists(output_path):
        print("loading from previous exists result...")
        with open(output_path, "r") as f:
            result = json.load(f)
            print(result)
    else:
        metrics = {"rouge-1":{"r":[], "p":[], "f":[]}, "rouge-2":{"r":[], "p":[], "f":[]}, "rouge-l":{"r":[], "p":[], "f":[]}, "bert-score":{"r":[], "p":[], "f":[]}}

        with open(method_path, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                task_id = data["task_id"]
                completion = data["completion"]
                base_complete = baseline[task_id]

                sub_metric = cal_metrics(completion[0], base_complete)

                for key in metrics:
                    try:
                        for sub_key in metrics[key].keys():
                            metrics[key][sub_key].append(sub_metric[key][sub_key])
                    except:
                        metrics[key].append(sub_metric[key])


        for key in metrics.keys():
            try:
                for sub_key in metrics[key].keys():
                    metrics[key][sub_key] = sum(metrics[key][sub_key]) / len(metrics[key][sub_key])
            except:
                metrics[key] = sum(metrics[key]) / len(metrics[key])
        print(metrics)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent="\t")
            f.close()

