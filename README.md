# Transformers

## Description:

This repo is a collection of PyTorch implementations of Transformer architectures with simple flexible config for ease of experimentation. The goal is learning and experimentation.

## Tests:

Tests can be run using `pytest` from the root directory. There are also online colabs that should test any new architecture added to the repo on shakespeare character prediction.

1. [basic transformer](https://colab.research.google.com/drive/1cNjbbiDqeHyjGFyMnuag9RykDKL2XKLp)
2. [MoE transformer](https://colab.research.google.com/drive/193oYMnTx8FdJDMj_NOgyOng6j9nQc7K_)
3. [Relative Attention](https://colab.research.google.com/drive/1QvtlYzUswKXf0POsEKQItZ9R9JFr38KQ#scrollTo=s4x21oqxNGI5)

As well as this each architecture and layer should be benchmarked for speed using:

1. [Transformer-benchmarks](https://colab.research.google.com/drive/1hb9V6ne42awHTxKvcI0vct1SNEO7rock)
2. [Runtime Comparison](https://colab.research.google.com/drive/1fCJ1FsDTUF9cQIGCeVOl-2XqPj0gjSrO)

## Resources:

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
3. [minGPT](https://github.com/karpathy/minGPT)
4. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
5. [d2l-vision-transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)
6. [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)