# kpe-bench

This is a benchmark for keyphrase extraction using LLMs. It uses the Krapivin dataset from [here](https://www.semanticscholar.org/paper/Large-Dataset-for-Keyphrase-Extraction-Krapivin-Marchese/85182ae8803b609baa8c08106af350af010f8777) which was obtaining from the huggingface dataset [midas/krapivin](https://huggingface.co/datasets/midas/krapivin). 

It is in development and being run, and will only be run locally and on cheap models because benchmarking is expensive when your input dataset is 27M tokens over 2.6k messages.

The benchmark itself is computing the pairwise similarity of the generated and ground truth keyphrases. It embeds keyphrases uses hungarian optimal matching on cosine similarity to the ground truth keyphrases because what I care about is semantic similarity not exact matches. 