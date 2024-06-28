import json
import time

import openai
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm
import os

class Get:
    def __init__(self):
        self.prompt = ""
    def calc(self,query,instruction="", temp=0,n=1,model=3.5,task="code"):
            openai.api_type = "azure"
            openai.api_base = "https://runway.devops.xiaohongshu.com"
            openai.api_version = "2023-05-15"  # "2023-03-15-preview"
            if model==3.5:
                openai.api_key = "please input your token"
                id = "gpt-35-turbo"
            elif model==4:
                openai.api_key = "please input your token"
                id = "gpt4-PTU"
            else:
                openai.api_key = "please input your token"
                id = "gpt-4"

            if task == "human_eval" or task == "mbpp":
                instruction = "Please complete the given function with Python code according to the description."
                cur = []
            elif task == "DM":
                instruction = "You are a helpful assistant. Please write a short, one-sentence summary of the following document:"
                task_prompts = [
                    {
                        "Q": "Document to be summarized:\nA northeast Ohio woman has been charged with felonious assault for allegedly stabbing her boyfriend after yelling at him for eating all of the couple’s salsa. A police report says 61-year-old Ronnie Buckner told officers he was eating salsa with his girlfriend, Phyllis Jefferson, at his Akron, Ohio, apartment Sunday when she started complaining that he was eating all of it. Buckner told police that 50-year-old Jefferson then started yelling and then stabbed him in his pelvis with a pen. Over salsa: Phyllis Jefferson, 60, is accused of stabbing her boyfriend repeatedly because he ate all their salsa . Jefferson allegedly then went to smash the television, but when Buckner grabbed it she took off for the kitchen and grabbed a knife. She's then accused of stabbing him in the left side of the stomach. Buckner was found holding his stomach and covered in blood outside his apartment when police arrived. Jefferson sped away, but police say they later stopped and arrested her on Interstate 77, reports Cleveland.com. She’s also charged with misdemeanor criminal endangering. Court records list no attorney for her. Buckner was hospitalized for his injuries, which appeared not to be life-threatening. Jefferson's boyfriend Ronnie Buckner was found holding his stomach and covered in blood outside his apartment when police arrived. His injuries appeared non life threatening.\nSummary:\n",
                        "A": "Phyllis Jefferson reportedly tried to smash his TV first."#"Phyllis Jefferson, 50, of Akron, Ohio reportedly tried to smash his TV first."
                    },
                    {
                        "Q": "Document to be summarized:\nToast Of New York has been ruled out of the Dubai World Cup at Meydan on March 31 after suffering a setback. A stunning winner of the UAE Derby on World Cup night last March, Jamie Osborne's stable star ended his three-year-old campaign with a fantastic effort in defeat when beaten just a nose by Bayern in the Breeders' Cup Classic at Santa Anita in November. Sheikh Joaan Al Thani's Al Shaqab Racing operation stepped in to buy the colt earlier this year, and Osborne had been preparing him for a tilt at the world's richest race where he would have been ridden by Frankie Dettori. Toast Of New York, pictured here with trainer Jamie Osborne, is set for a spell on the sidelines . Toast Of New York is, however, now set for a spell on the sidelines. The trainer said: 'He's had a setback and, sadly, he's going to miss the World Cup. I have to speak to the Al Shaqab team and I'm sure we'll be formulating another plan. 'It's very disappointing for Sheikh Joaan and his team, and obviously for everybody here but he'll be back.\nSummary:\n",
                        "A": "Toast of New York ruled out of Dubai World Cup at Meydan on March 31." #"Jamie Osborne's stable star set for a spell on the sidelines."
                    },
                    {
                        "Q": "Document to be summarized:\nThis is the adorable moment a hungry raccoon exercises a degree of caution as he's offered a treat. Video footage, viewed more than two million times, shows the furry animal hiding in a hollowed-out out tree trunk. Then, as a piece of food is presented, he cautiously reaches his small paws forwards. With his eyes on the prize, he delicately clutches the nibble tight. Bystanders are heard laughing in the background as they watch the timid critter in action. Once he's got his treat, the raccoon retreats back into his hidey-hole. The heartwarming clip was filmed by Russian YouTube user, lgreko100. It was captured at an unknown location last year and recently resurfaced. Dinner is served: This is the adorable moment a hungry raccoon exercises a degree of caution as he's offered a treat . Come on: Video footage, viewed more than two million times, shows the furry animal hiding in a hollowed-out out tree trunk . Thank you! Then, as a piece of food is presented, he cautiously reaches his small paws forwards. So long: Once he's got his treat, the raccoon retreats back into his hidey-hole.\nSummary:\n",
                        "A": "The heartwarming clip was filmed by Russian YouTube user, lgreko100."
                    },
                    {
                        "Q": "Document to be summarized:\nGetting children to welcome a new baby into the home can sometimes prove difficult - especially when it wasn't the playmate they were hoping for. Indeed, one mother from Colorado filmed the moment she told her daughter she was expecting a boy. Footage shows the toddler's smile quickly turning to a frown and then a cry 'no' as she breaks down in tears. 'I want a baby sister!' the little girl dramatically sobs. Her mother calmly explains that it isn't possible and she'll be getting a baby brother. 'That's why it's blue,' she says referencing a baby shower cupcake at home. But the girl remains vehemently opposed to the idea. The clip ends with her exclaiming: 'I want a baby sister right now!' Hopefully she eventually came around to the idea of having a little brother. Caught on camera: One mother from America filmed the moment she told her daughter she was expecting a boy - the infant wasn't too impressed . Chain reaction: The toddler's smile quickly turns to a frown and then a cry 'no' as she breaks down in tears.\nSummary:\n",
                        "A": "A mother from America filmed the scene play out." # "Her daughter   immediately started crying when she found out she was going to get a brother."
                    }
                ]
                cur = []
                for tem in task_prompts:
                    cur.append({"role": "user", "content": tem["Q"]})
                    cur.append({"role": "assistant", "content": tem["A"]})
                query = "Document to be summarized:\n " + query + "\n" + "Summary:\n"

            else:
                cur = []
                instruction = instruction

            count = 0
            while True:
                messages = [{"role": "system", "content": instruction}] + cur + [{"role": "user", "content": query}]
                try:
                    response = openai.ChatCompletion.create(
                        deployment_id=id,
                        messages=messages,
                        top_p=0.9,
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




if __name__ == "__main__":
    Gen = Get()
    model = 4
    temp = 0.8
    n = 50
    samples = []
    i = 0
    dataset = ["mbpp", "human_eval", "summ_screen", "DM"]

    dataset = dataset[0]

    if dataset == "human_eval":
        for task_id, problem in tqdm(get_human_eval_plus().items()):
            samples.append(dict(task_id=task_id, completion=Gen.calc(query=problem["prompt"], temp=temp, n=n, model=model, task=dataset)[:n]))
            write_jsonl("humaneval_gpt_{}_samples_n_{}_temp_{}.jsonl".format(model, n, temp), samples)

        write_jsonl("humaneval_gpt_{}_samples_n_{}_temp_{}.jsonl".format(model, n, temp), samples)
    if dataset == "mbpp":
        for i in range(1, 10):
            samples = []
            for task_id, problem in tqdm(get_mbpp_plus().items()):
                if task_id == "Mbpp/77" or task_id == "Mbpp/125" or task_id == "Mbpp/598":
                    samples.append(dict(task_id=task_id, completion=Gen.calc(query=problem["prompt"], temp=temp, n=n, model=model, task=dataset)[:n]))
                    write_jsonl("mbpp_gpt_{}_samples_n_{}_temp_{}_madeup_{}.jsonl".format(model, n, temp, i), samples)

            write_jsonl("mbpp_gpt_{}_samples_n_{}_temp_{}_madeup_{}.jsonl".format(model, n, temp, i), samples)

    if dataset == "summ_screen":
        input_path = "SummScreen.json"
        with open(input_path, "r") as f:
            data = json.load(f)
            f.close()
        for item in tqdm(data):
            samples.append(dict(input=item["input"], answer=item["output"], completion=Gen.calc(query=item["input"], temp=temp, n=n, model=model, task=dataset)[:n]))
            write_jsonl("summ_gpt_{}_samples_n_{}_temp_{}_1.jsonl".format(model, n, temp), samples)
        write_jsonl("summ_gpt_{}_samples_n_{}_temp_{}_1.jsonl".format(model, n, temp), samples)

    if dataset == "DM":
        input_path = "DM.json"
        with open(input_path, "r") as f:
            data = json.load(f)
            f.close()
        for item in tqdm(data[:1000]):
            samples.append(dict(input=item["input"], answer=item["output"],
                                completion=Gen.calc(query=item["input"], temp=temp, n=n, model=model, task=dataset)[:n]))
            write_jsonl("dm_gpt_{}_samples_n_{}_temp_{}_new.jsonl".format(model, n, temp), samples)
        write_jsonl("dm_gpt_{}_samples_n_{}_temp_{}_new.jsonl".format(model, n, temp), samples)
