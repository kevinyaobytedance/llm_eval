# Copyright (C) 2023 ByteDance. All Rights Reserved.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import re
import torch

import openai
import jsonlines
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(30))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(30))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def dump_to_jsonlines(data, fn):
    # Dump to jsonline.
    with jsonlines.open(fn, mode='w') as writer:
        for item in data:
            writer.write(item)
    return


def query_llm(prompt, model, temperature=0.7, max_tokens=50):
    assert model in ['text-davinci-003', 'davinci', 'gpt-3.5-turbo',
                     'gpt-4']
    # Completion models.
    if model in ['text-davinci-003', 'davinci']:
        answer = completion_with_backoff(
                engine=model, prompt=prompt,
                temperature=temperature, max_tokens=max_tokens, logprobs=1)
        answer = answer['choices'][0]['text']
    # Chat models.
    elif model in ['gpt-3.5-turbo', 'gpt-4']:
        answer = chat_completion_with_backoff(
                    model=model,
                    messages=[
                            {"role": "user", "content": prompt},
                                ],
                    temperature=temperature, max_tokens=max_tokens)
        answer = answer['choices'][0]['message']['content']

    return answer.strip()


def parse_keyword_list(text):
    keywords = text.strip().split('\n')
    pattern = r"\d+\.\s(.*)"
    extracted_keywords = []
    for keyword in keywords:
        match = re.search(pattern, keyword)
        if match:
            extracted_keywords.append(match.group(1))

    return extracted_keywords


def query_opt(prompt, generator, greedy_sampling=False):
    if greedy_sampling:
        answer = generator(prompt, do_sample=True, top_k=1)
    else:
        answer = generator(prompt)
    answer = answer[0]['generated_text']
    
    # Remove the beginning of answer if duplicate with the prompt.
    if answer.startswith(prompt):
        answer = answer[len(prompt):]

    return answer


def query_flan_t5(prompt, model, tokenizer,
                  greedy_sampling=False, max_length=20):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    if greedy_sampling:
        outputs = model.generate(
            input_ids, do_sample=True, temperature=0.01, max_length=max_length)
    else:
        outputs = model.generate(input_ids, max_length=max_length)
    
    answer = tokenizer.decode(outputs[0])

    # Replace tags.
    answer = answer.replace("<pad>", "").replace("</s>", "").strip()

    return answer