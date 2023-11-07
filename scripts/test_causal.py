import openai
import glob
import json
import os
import re
import numpy as np
import csv
import argparse
import random
import torch
from utils import *
from transformers import pipeline, set_seed, T5Tokenizer, T5ForConditionalGeneration


def extract_letter_answers(text):
    pattern = r'\b[A|B]\)'
    matches = re.findall(pattern, text)
    answers = [match.strip(')').strip() for match in matches]
    return answers

def extract_yes_no(text):
    pattern = r'\b[Yy][Ee][Ss]\b|\b[Nn][Oo]\b'
    matches = re.findall(pattern, text)
    answers = []
    for ele in matches:
        if ele.lower() == "yes":
            answers.append("A")
        else:
            answers.append("B")
    return answers

def extract_first_yes_no(text):
    pattern = r'\b[Yy][Ee][Ss]\b|\b[Nn][Oo]\b'
    matches = re.findall(pattern, text)
    answers = []
    for ele in matches:
        if ele.lower() == "yes":
            answers.append("A")
        else:
            answers.append("B")
    if len(answers) == 0:
        return ["C"]
    return [answers[0]]

def write_acc(true_answers, answer_list, wrt):
    # for model0
    N = len(true_answers)
    q1_correct = 0.
    q2_correct = 0.
    q1_understood = 0.
    q2_understood = 0.
    for i in range(N):
        q1_correct += true_answers[i][0] == answer_list[i][0]
        q2_correct += true_answers[i][1] == answer_list[i][1]
        q1_understood += answer_list[i][0] in ["A","B"]
        q2_understood += answer_list[i][1] in ["A","B"]

    q1_acc = np.round(q1_correct / float(N),4)
    q2_acc = np.round(q2_correct / float(N),4)

    q1_understood_rate = np.round(q1_understood / float(N),4)
    q2_understood_rate = np.round(q2_understood / float(N),4)

    wrt.writerow([q1_acc,q2_acc,q1_understood_rate,q2_understood_rate])

# Enter your own OpenAI API key.
openai.api_key = ''

def find_word_after_keyword(keyword, text):
    words = text.split()
    if keyword in words:
        index = words.index(keyword)
        if index < len(words)-1:
            return words[index+1]
    return None

def get_q(template, event_a, event_b):
    q = template.replace('[Event_A]', event_a.replace(".",""))
    q = q.replace('[Event_B]', event_b.replace(".",""))
    q = q.strip()
    return q

def get_q_one_shot(template, event_a, event_b, event_c, event_d, answer):
    # try to use one-shot ICL for force opt to answer A) or B)

    q = template.replace('[Event_A]', event_a.replace(".",""))
    q = q.replace('[Event_B]', event_b.replace(".",""))
    q = q.replace('[Event_C]', event_c.replace(".",""))
    q = q.replace('[Event_D]', event_d.replace(".",""))
    q = q.replace('[Answer]', answer.replace(".",""))

    q = q.strip()
    return q

def load_csv(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        ls = [row for row in reader]
        return ls


def save_answer(model_name, answer_list, answer_raw_list, true_answers, answer_path, answer_raw_path, res_path, MODELS, root):
    if model_name in MODELS and len(answer_list) > 0:
        with open(answer_path, 'w') as csvfile:
            wrt = csv.writer(csvfile)
            wrt.writerows(answer_list)
        
        with open(answer_raw_path, "w") as json_file:
            json.dump(answer_raw_list, json_file, indent=4)
        
        wf = open(os.path.join(root,res_path), 'w')
        res_wrt = csv.writer(wf)
        write_acc(true_answers, answer_list, res_wrt)


def main(MODELS,OPT,FLAN):
    root = 'eval_results/causal/test2'

    template_q1 = open("causal_questions/prompt_q1.txt","r").read()
    template_q2 = open("causal_questions/prompt_q2.txt","r").read()

    # template_q1_opt = open("causal_questions/prompt_q1_opt.txt","r").read()
    # template_q2_opt = open("causal_questions/prompt_q2_opt.txt","r").read()

    with open('causal_questions/data2.json', 'r') as f:
        data = json.load(f)["data"]
    
    # load models
    # Init Huggingface model.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # OPT-1.3B.
    if 'opt-1.3b' in MODELS and OPT:
        opt_generator = pipeline('text-generation', model="facebook/opt-1.3b",
                                do_sample=True, max_length=200, device=device)
    else:
        opt_generator = None
    
    # flan-t5-xxl.
    if 'flan-t5-xxl' in MODELS and FLAN:
        flan_t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl", device_map="auto")
        flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    else:
        flan_t5_model = None
        flan_t5_tokenizer = None

    true_answers = []

    answer0_list = []
    answer1_list = []
    answer2_list = []
    answer3_list = []
    answer_flan_list = []
    answer_opt_list = []

    answer0_raw_list = {}
    answer1_raw_list = {}
    answer2_raw_list = {}
    answer3_raw_list = {}
    answer_flan_raw_list = {}
    answer_opt_raw_list = {}

    answer0_path = os.path.join(root,'answers0.csv') #GPT-3.5
    answer1_path = os.path.join(root,'answers1.csv') #Davinci-003
    answer2_path = os.path.join(root,'answers2.csv') #Davinci
    answer3_path = os.path.join(root,'answers3.csv') #GPT-4

    answer_flan_path = os.path.join(root,'answers_flan.csv') #Flan
    answer_opt_path = os.path.join(root,'answers_opt.csv') #OPT

    answer0_raw_path = os.path.join(root,'answers0_raw.csv')
    answer1_raw_path = os.path.join(root,'answers1_raw.csv')
    answer2_raw_path = os.path.join(root,'answers2_raw.csv')
    answer3_raw_path = os.path.join(root,'answers3_raw.csv')

    answer_flan_raw_path = os.path.join(root,'answers_flan_raw.csv') #Flan
    answer_opt_raw_path = os.path.join(root,'answers_opt_raw.csv') #OPT

    # flag0 = True
    # flag1 = True

    if args.reuse:
        if os.path.exists(answer0_path) and 'gpt-3.5-turbo' in MODELS:
            answer0_list = load_csv(answer0_path)
            if len(answer0_list) == 0:
                raise ValueError("empty csv file {}".format(answer0_path))
    
        if os.path.exists(answer1_path) and 'text-davinci-003' in MODELS:
            answer1_list = load_csv(answer1_path)
            if len(answer1_list) == 0:
                raise ValueError("empty csv file {}".format(answer1_path))

        if os.path.exists(answer2_path) and 'davinci' in MODELS:
            answer2_list = load_csv(answer2_path)
            if len(answer2_list) == 0:
                raise ValueError("empty csv file {}".format(answer2_path))

        if os.path.exists(answer_flan_path) and 'flan-t5-xxl' in MODELS:
            answer_flan_list = load_csv(answer_flan_path)
            if len(answer_flan_list) == 0:
                raise ValueError("empty csv file {}".format(answer_flan_path))

        if os.path.exists(answer_opt_path) and 'opt-1.3b' in MODELS:
            answer_opt_list = load_csv(answer_opt_path)
            if len(answer_opt_list) == 0:
                raise ValueError("empty csv file {}".format(answer_opt_path))

    N = len(data)

    answers = [["B","B"],["A","A"],["A","B"]]
    for i, row in enumerate(data):

        event_a, event_b, rela_type = row
        assert(rela_type in [1,2,3])
        ans_1, ans_2 = answers[rela_type-1]

        # event_a, event_b, ans_1, ans_2 = row
        # randomly draw another data point

        # j = random.randint(0, N-1)
        # while i==j:
        #     j = random.randint(0, N-1)
        
        # event_c, event_d, ans_3, ans_4 = data[j]

        # if ans_1 == "A" and ans_2 == "B":
        #     continue
        true_answers.append([ans_1,ans_2])

        q1 = get_q(template_q1, event_a, event_b)
        q2 = get_q(template_q2, event_a, event_b)

        # q1_opt = get_q(template_q1_opt, event_a, event_b)
        # q2_opt = get_q(template_q2_opt, event_a, event_b)

        # q1_opt = get_q_one_shot(template_q1_opt, event_a, event_b, event_c, event_d, ans_3)
        # q2_opt = get_q_one_shot(template_q2_opt, event_a, event_b, event_c, event_d, ans_4)

        if args.reuse:
            continue

        # Test on all 3 models.
        for model in MODELS:
            if model == 'opt-1.3b':
                # q1 = q1 + '\n Your answer is _.'
                # q2 = q2 + '\n Your answer is _.'

                # answer_q1_opt_raw = query_opt(q1_opt, opt_generator)
                # answer_q2_opt_raw = query_opt(q2_opt, opt_generator)
                # answer_q1_opt = extract_first_yes_no(answer_q1_opt_raw)
                # answer_q2_opt = extract_first_yes_no(answer_q2_opt_raw)

                answer_q1_opt_raw = query_opt(q1, opt_generator)
                answer_q2_opt_raw = query_opt(q2, opt_generator)
                answer_q1_opt = extract_letter_answers(answer_q1_opt_raw)
                answer_q2_opt = extract_letter_answers(answer_q2_opt_raw)

                if len(answer_q1_opt) != 1:
                    # did not understand the question
                    answer_q1_opt = ["C"]
                
                if len(answer_q2_opt) != 1:
                    # did not understand the question
                    answer_q2_opt = ["C"]

                answer_opt_list.append([answer_q1_opt[0],answer_q2_opt[0],
                                         answer_q1_opt[0]==ans_1, answer_q2_opt[0]==ans_2])
                
                answer_opt_raw_list[i] = {"q1":q1, "q2":q2, 
                "ans_1": answer_q1_opt[0], "ans_2": answer_q2_opt[0],
                "ans_1_true": ans_1, "ans_2_true": ans_2,
                "ans_1_raw":answer_q1_opt_raw, "ans_2_raw":answer_q2_opt_raw}


            if model == 'flan-t5-xxl':
                answer_q1_flan_raw = query_flan_t5(
                                q1, flan_t5_model, flan_t5_tokenizer)
                answer_q2_flan_raw = query_flan_t5(
                                q2, flan_t5_model, flan_t5_tokenizer)
                
                answer_q1_flan = extract_letter_answers(answer_q1_flan_raw)
                answer_q2_flan = extract_letter_answers(answer_q2_flan_raw)

                if len(answer_q1_flan) != 1:
                    # did not understand the question
                    answer_q1_flan = ["C"]

                if len(answer_q2_flan) != 1:
                    # did not understand the question
                    answer_q2_flan = ["C"]
                
                answer_flan_list.append([answer_q1_flan[0],answer_q2_flan[0], 
                                         answer_q1_flan[0]==ans_1, answer_q2_flan[0]==ans_2])
                
                answer_flan_raw_list[i] = {"q1":q1, "q2":q2, 
                "ans_1": answer_q1_flan[0], "ans_2": answer_q2_flan[0],
                "ans_1_true": ans_1, "ans_2_true": ans_2,
                "ans_1_raw":answer_q1_flan_raw, "ans_2_raw":answer_q2_flan_raw}

            if model == 'gpt-4':
                # q1 = q1 + '\nAnswer the question with A) or B).'
                # q2 = q2 + '\nAnswer the question with A) or B).'

                answer3_q1_raw = query_llm(q1, model, temperature=0.7, max_tokens=50)
                answer3_q2_raw = query_llm(q2, model, temperature=0.7, max_tokens=50)

                answer3_q1 = extract_letter_answers(answer3_q1_raw)
                answer3_q2 = extract_letter_answers(answer3_q2_raw)

                if len(answer3_q1) != 1:
                    # did not understand the question
                    answer3_q1 = ["C"]
                if len(answer3_q2) != 1:
                    # did not understand the question
                    answer3_q2 = ["C"]

                answer3_list.append([answer3_q1[0],answer3_q2[0], answer3_q1[0]==ans_1, answer3_q2[0]==ans_2])

                answer3_raw_list[i] = {"q1":q1, "q2":q2, 
                "ans_1": answer3_q1[0], "ans_2": answer3_q2[0],
                "ans_1_true": ans_1, "ans_2_true": ans_2,
                "ans_1_raw":answer3_q1_raw, "ans_2_raw": answer3_q2_raw}

            if model == 'gpt-3.5-turbo':
                # q1 = q1 + '\nAnswer the question with A) or B).'
                # q2 = q2 + '\nAnswer the question with A) or B).'

                answer0_q1_raw = query_llm(q1, model, temperature=0.7, max_tokens=50)
                answer0_q2_raw = query_llm(q2, model, temperature=0.7, max_tokens=50)

                answer0_q1 = extract_letter_answers(answer0_q1_raw)
                answer0_q2 = extract_letter_answers(answer0_q2_raw)

                if len(answer0_q1) != 1:
                    # did not understand the question
                    answer0_q1 = ["C"]
                if len(answer0_q2) != 1:
                    # did not understand the question
                    answer0_q2 = ["C"]

                answer0_list.append([answer0_q1[0],answer0_q2[0], answer0_q1[0]==ans_1, answer0_q2[0]==ans_2])
                answer0_raw_list[i] = {"q1":q1, "q2":q2, 
                "ans_1": answer0_q1[0], "ans_2": answer0_q2[0],
                "ans_1_true": ans_1, "ans_2_true": ans_2,
                "ans_1_raw":answer0_q1_raw, "ans_2_raw": answer0_q2_raw}

            if model == 'text-davinci-003':

                # q1 = q1 + '\nAnswer the question with A) or B).'
                # q2 = q2 + '\nAnswer the question with A) or B).'
                answer1_q1_raw = query_llm(q1, model, temperature=0.7, max_tokens=50)
                answer1_q2_raw = query_llm(q2, model, temperature=0.7, max_tokens=50)

                answer1_q1 = extract_letter_answers(answer1_q1_raw)
                answer1_q2 = extract_letter_answers(answer1_q2_raw)

                if len(answer1_q1) != 1:
                    # did not understand the question
                    answer1_q1 = ["C"]
                if len(answer1_q2) != 1:
                    # did not understand the question
                    answer1_q2 = ["C"]

                answer1_list.append([answer1_q1[0],answer1_q2[0], answer1_q1[0]==ans_1, answer1_q2[0]==ans_2
                                     ])
                answer1_raw_list[i] = {"q1":q1, "q2":q2, 
                "ans_1": answer1_q1[0], "ans_2": answer1_q2[0],
                "ans_1_true": ans_1, "ans_2_true": ans_2,
                "ans_1_raw":answer1_q1_raw, "ans_2_raw": answer1_q2_raw}

            if model == 'davinci':
                # Need a special QA format for the original GPT-3 model.
                # q1 = q1 + '\nChoose A) or B) only for each task. Answers are __.\n\nAnswer:'
                # q2 = q2 + '\nChoose A) or B) only for each task. Answers are __.\n\nAnswer:'

                # q1 += '\n Answers are __.\n\nAnswer:'
                # q2 += '\n Answers are __.\n\nAnswer:'

                answer2_q1_raw = query_llm(q1, model, temperature=0.7, max_tokens=50)
                answer2_q2_raw = query_llm(q2, model, temperature=0.7, max_tokens=50)

                answer2_q1 = extract_letter_answers(answer2_q1_raw)
                answer2_q2 = extract_letter_answers(answer2_q2_raw)

                if len(answer2_q1) != 1:
                    # did not understand the question
                    answer2_q1 = ["C"]
                if len(answer2_q2) != 1:
                    # did not understand the question
                    answer2_q2 = ["C"]

                answer2_list.append([answer2_q1[0],answer2_q2[0], answer2_q1[0]==ans_1, answer2_q2[0]==ans_2])
                answer2_raw_list[i] = {"q1":q1, "q2":q2, 
                "ans_1": answer2_q1[0], "ans_2": answer2_q2[0],
                "ans_1_true": ans_1, "ans_2_true": ans_2,
                "ans_1_raw":answer2_q1_raw, "ans_2_raw": answer2_q2_raw}
            # else:
            #     print('Wrong model')
            #     return      
        print('Finished question %d/%d ' % (i, len(data)))   

    # save the answers

    save_answer('gpt-3.5-turbo', answer0_list, answer0_raw_list, true_answers, answer0_path, answer0_raw_path, 'final_results0.csv', MODELS, root)
    save_answer('text-davinci-003', answer1_list, answer1_raw_list, true_answers, answer1_path, answer1_raw_path , 'final_results1.csv', MODELS, root)
    save_answer('davinci', answer2_list, answer2_raw_list, true_answers, answer2_path, answer2_raw_path, 'final_results2.csv', MODELS, root)
    save_answer('gpt-4', answer3_list, answer3_raw_list ,true_answers, answer3_path, answer3_raw_path, 'final_results3.csv', MODELS, root)
    save_answer('opt-1.3b', answer_opt_list, answer_opt_raw_list, true_answers, answer_opt_path, answer_opt_raw_path, 'final_results_opt.csv', MODELS, root)
    save_answer('flan-t5-xxl', answer_flan_list, answer_flan_raw_list, true_answers, answer_flan_path, answer_flan_raw_path, 'final_results_flan.csv', MODELS, root)
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--reuse", action='store_true')

    args, _ = parser.parse_known_args()
    OPT = False
    FLAN = False

    if args.model == "opt":
        MODELS = ['opt-1.3b']
        OPT = True

    elif args.model == "flan":
        MODELS = ['flan-t5-xxl']
        FLAN = True
    
    elif args.model == "gpt-4":
        MODELS = ['gpt-4']

    elif args.model == "chatgpt":
        MODELS = ['gpt-3.5-turbo']
    
    elif args.model == "text-davinci-003":
        MODELS = ["text-davinci-003"]
    
    elif args.model == "davinci":
        MODELS = ["davinci"]

    else:
        raise ValueError("make sure --model is correct.")

    main(MODELS,OPT,FLAN)
    