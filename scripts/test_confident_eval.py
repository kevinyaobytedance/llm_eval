# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import re
import os


def create_dataframe(file_path):
    data = []

    with open(file_path, 'r') as file:
        lines = file.read().split('---')

    for entry in lines:
        entry_lines = entry.strip().split('\n')
        if len(entry_lines) < 3:  # Not a valid entry
            continue

        data_dict = {}
        for line in entry_lines:
            if line.startswith('Question:'):
                data_dict['Question'] = line.replace('Question:', '').strip()
            elif line.startswith('Answer:'):
                data_dict['Answer'] = line.replace('Answer:', '').strip()
            elif line.startswith('P: !!'):
                probabilities = line.replace(
                    'P: !!', '').replace(
                    '!!', '').strip()
                data_dict['Probabilities'] = np.fromstring(
                    probabilities[1:-1], sep=', ')

        data.append(data_dict)

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(data)

    return df


def process_data(file_path):
    diffs_final = []
    probs_vec_final = []
    verbal_vec_final = []
    for part in range(6):
        print(part)
        with open(file_path + f'davinci_3_with_prob_FAIR_partare_{part}.txt', 'r') as file:
            data = file.read()

        sections = data.split("----")
        diffs = []
        probs_vec = []
        verbal_vec = []
        counter = 0
        full_counter = 0
        answer_a_vec = []
        for section in sections:
            if section.strip() == "":
                continue

            answer_line = re.findall(
                r"\n\nAnswer: ANS: A, A: (\d+)%\, B: (\d+)%\, C: (\d+)%", section)
            prob_line = re.findall(
                r"P: !!\((.*), (.*), (.*), (.*)\)!!", section)

            if not answer_line or not prob_line:
                continue

            answer_values = [int(x) / 100 for x in answer_line[0]]
            # Only take the first 3 probabilities
            prob_values = [float(x) for x in prob_line[0][:3]]
            diff = [a - p for a, p in zip(answer_values, prob_values)]
            full_counter += 1
            answer_a_vec.append(prob_values[0])

            if abs(np.array(diff)[0]) > 0.05:
                counter += 1
            if abs(np.array(diff)[1]) > 0.05:
                counter += 1
            if abs(np.array(diff)[2]) > 0.05:
                counter += 1
            diffs.extend(diff)
            probs_vec.extend(answer_values)
            verbal_vec.extend(prob_values)
        diffs_final.extend(diffs)
        probs_vec_final.extend(probs_vec)
        verbal_vec_final.extend(verbal_vec)
    return np.mean(np.abs(diffs_final)),np.std(np.abs(diffs_final)), \
            counter, full_counter, diffs_final, answer_a_vec, \
            probs_vec_final, verbal_vec_final


if __name__ == "__main__":
    file_path = 'eval_results/fairness/gender/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    mean_, std_, counter, full_counter, diffs, answer_a_vec, probs_vec_final, \
            verbal_vec_final = process_data(file_path)
    plt.hist(diffs, bins=50)
    plt.savefig('hist.png')
