from datasets import load_dataset
import json
import ast

answer_letter_map = {}
letter_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']):
    answer_letter_map[i] = letter

def create_eval_data():

    bbeh = load_dataset("BBEH/bbeh", split="train")

    bbeh_data = []
    for i, item in enumerate(bbeh):
        question = item["input"]
        answer = item["target"]
        if answer[0] == "(":
            bbeh_data.append({
                "id": f"bbeh_{i}",
                "question": question,
                "answer": answer,
            })

    with open("data/bbeh.json", "w") as f:
        json.dump(bbeh_data, f, indent=4)

    dataset = load_dataset("WildEval/ZebraLogic", 'mc_mode', split='test')
    zebra_logic_data = []

    for data in dataset:

        choices = data['choices']
        choices_str = "\n".join([f"Option {letter_array[i]}: {choice}" for i, choice in enumerate(choices)])


        question = f"""We aim to answer a question based on the following zebra puzzle:

{data['puzzle']}
Answer the following question based on the puzzle.
Question: {data['question']}

{choices_str}"""
        zebra_logic_data.append({
            'id': data['id'],
            'question': question,
            'answer': letter_array[choices.index(data['answer'])],
        })

    with open('data/zebralogicbench.json', 'w') as f:
        json.dump(zebra_logic_data, f, indent=4)
    
    dataset = load_dataset("TAUR-Lab/MuSR")

    musr_data = []

    for i, data in enumerate(dataset['murder_mysteries']):
        narrative = data['narrative']
        question = data['question']
        choices = data['choices']
        choices = ast.literal_eval(choices)

        choices_str ="\n".join([f"Option {answer_letter_map[i]}: {choice}" for i, choice in enumerate(choices)])


        prompt = f"""You are given a narrative and a question. The question is a multiple choice question with multiple options. Choose the correct answer from the options based on the information provided.

Narrative:
{narrative}

Question:
{question}

{choices_str}"""

        musr_data.append({
            'id': f'musr_murder_mysteries_{i}',
            'question': prompt,
            'answer': letter_array[int(data['answer_index'])],
        })

    with open('data/musr.json', 'w') as f:
        json.dump(musr_data, f, indent=4)


    dataset = load_dataset("yale-nlp/FOLIO")['validation']

    folio_data = []

    label_map = {
        'True': 'A',
        'Uncertain': 'B',
        'False': 'C'
    }

    for i, data in enumerate(dataset):
        premises = data['premises']
        conclusion = data['conclusion']

        prompt = f"""You are provided with a set of logical premises and a conclusion. Determine whether the conclusion is True, Uncertain, or False.

    Premises:
    {premises}

    Conclusion:
    {conclusion}

    Option A: True
    Option B: Uncertain
    Option C: False"""
        folio_data.append({
            'id': f'folio_{i}',
            'question': prompt,
            'answer': label_map[data['label']],
        })

    with open('data/folio.json', 'w') as f:
        json.dump(folio_data, f, indent=4)

    


if __name__ == "__main__":
    create_eval_data()