## Code for "[Trustworthy LLMs: a Survey and Guideline for Evaluations and Alignments](https://arxiv.org/abs/2308.05374)"

`scripts/`: Scripts for testing LLM trustworthiness in Section 11 of the paper.
- `test_hallucination.py`: Test LLM Hallucination (Section 11.2).
- `test_safety.py`: Test the safety of LLM responses (Section 11.3).
- `test_fairness.py`: Test the fairness of LLM responses (Section 11.4).
- `test_confident_eval_fair.py` and `test_confident_eval.py`: Test miscalibration of LLMs' confidence (Section 11.5).
- `test_misuse.py`: Test the resistence to misuse in LLMs (Section 11.6).
- `test_copytight.py`: Test the copyright-protected data leakage in LLMs (Section 11.7).
- `test_causal.py`: Test LLMs' causal reasoning ability (Section 11.8).
- `test_typo.py`: Test LLMs' robustness against typo attacks (Section 11.9).

`gen_data/`: Generated data and results from our testing.
- Note that we omit copyright results because it contains copyright-protected text.

`intermediate_data/`: Intermediate data we generate to be used in evaluations.

Citation:
```
@inproceedings{liu2023trustllm, 
title={​Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment}, 
author={​Liu, Yang and Yao, Yuanshun Yao and Ton, Jean-Francois Ton and Zhang, Xiaoying and Guo, Ruocheng and Klochkov, Yegor and Taufiq, Muhammad Faaiz and Li, Hang}, 
booktitle={preprint}, 
year={2023}
}
```