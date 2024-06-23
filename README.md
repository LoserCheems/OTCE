# OTCE

Read this in [English](README.md)

![OTCE](./assets/otce.png)
![selective_state_space_model_with_positional_encoding](./assets/selective_state_space_model_with_positional_encoding.png)
![quadratic_self_attention_with_positional_encoding](./assets/quadratic_self_attention_with_positional_encoding.png)
![cross_domain_moe](./assets/cross_domain_moe.png)

> **OTCE: Hybrid SSM and Attention with Cross Domain Mixture of Experts to construct Observer-Thinker-Conceiver-Expresser**\
> Jingze Shi*\
> Paper: 


## About

OTCE is a hybrid of SSM and Attention algorithms, with a sparse model architecture with cross-domain shared parameters, which outperforms models driven solely by SSM or Attention in language modeling.

As a poor student, most of the computing power of this project and a small amount of data come from the medical engineering cross-project resources of my school. Checkpoint weights are not allowed to be open-sourced, only the modeling code written by me can be open-sourced.


## Requirements

- Linux
- NVIDIA GPU
- CUDA 11.6+
- PyTorch 1.12+
- `pip install transformers causal-conv1d>=1.2.0 mamba-ssm sentencepiece`

## Usage

Same as using tokenizers, configurations, and model methods in the Transformers library.