import json
import argparse
from evalplus.eval import (
SUCCESS,
compatible_eval_result,
estimate_pass_at_k,
untrusted_check,
)
import numpy as np
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--fusion_num", type=int, required=True)
    args = parser.parse_args()
    input_path = args.input_path
    start = [0, 30, 60, 90]
    start = args.start
    fusion_num = args.fusion_num
        
    with open(input_path, "r") as f:
        results = json.load(f)
        f.close()
    
    base_correct = []
    new_correct = []

    total = np.array([fusion_num for r in results["eval"].values()])
    for res in results["eval"].values():
        bc = sum([r[0] == SUCCESS for r in res["base"][start:start+fusion_num]])
        base_correct.append(bc)
        if res["plus"]:
            new_correct.append(sum([res["plus"][i][0] == res["base"][i][0] == SUCCESS for i in range(start, start+fusion_num)]))

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, np.array(base_correct), k).mean()
        for k in [1, 5, 10, 100]
        if total.min() >= k
    }
    print("Base")
    print(pass_at_k)

    if new_correct:
        print("Base + Extra")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 5, 10, 100]
            if (total >= k).all()
        }
        print(pass_at_k)
