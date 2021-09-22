![Language](https://img.shields.io/badge/language-Python-blue.svg)
![Stars](https://img.shields.io/github/stars/iwera-git/deepBioWSD?color=r)
![Repo Size](https://img.shields.io/github/repo-size/iwera-git/deepBioWSD?color=tomato)

<br>
<img align="left" src="imgs/dna_logo.png" width="110"> 

## deepBioWSD: Effective Deep Neural Word Sense Disambiguation of Biomedical Text Data

### Background
With the recent advances in biomedicine, we have a wealth of information hidden in unstructured narratives such as research articles and clinical documents. A high accuracy [Word Sense Disambiguation (WSD)](https://en.wikipedia.org/wiki/Word-sense_disambiguation) algorithm can avoid a myriad of downstream difficulties in the natural language processing (NLP) applications pipeline when we try to mine and exploit this data properly. This is mainly due to the fact that word sense ambiguity is a pervasive characteristic of a natural language; for example, the word _cold_ has several senses and may refer to _a disease_, _a temperature sensation_, or _an environmental condition_. The specific sense intended is determined by the textual context in which an instance of the ambiguous word appears. In **"I am taking aspirin for my cold"** the _disease_ sense is intended, in **"Let's go inside, I'm cold"** the _temperature sensation_ sense is meant, while **"It's cold today, only 2 degrees"**, implies the _environmental condition_ sense. Therefore, automatically identifying the intended sense of ambiguous words improves the proper inference of biomedical text data for clinical and biomedical applications. 

## deepBioWSD Network
This project addresses the substantial problem of WSD in NLP by introducing and developing a novel deep Bidirectional Long Short-Term Memory (BLSTM) network. We evaluate accuracy of our BLSTM network for the task of word sense disambiguation in the biomedical domain. First, we initialize the BLSTM network using pre-trained concept vectors (also known as concept embeddings). Then, we train the network on the biomedical textual data. As to the calculation of the pre-trained _concept embeddings_, we make use of [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/) and MEDLINE abstracts and also employ [Pointwise Mutual Information (PMI)](https://en.wikipedia.org/wiki/Pointwise_mutual_information) and [Latent Semantic Analysis/Indexing (LSA/LSI)](https://en.wikipedia.org/wiki/Latent_semantic_analysis). Finally, we test the converged model on a holdout set. The experimental result on the [MSH-WSD dataset](https://wsd.nlm.nih.gov/collaboration.shtml) ([MeSH WSD dataset](https://wsd.nlm.nih.gov/collaboration.shtml) from [National Library of Medicine, NLM](https://www.nlm.nih.gov/)) represents that the introduced deep learning model outperforms the state-of-the-art methods in terms of accuracy results.

## Project Outcome
The outcome of this project is directly applicable to a wide range of NLP applications. These applications run the gamut from [machine translation](https://en.wikipedia.org/wiki/Machine_translation) as well as [automatic text summarization](https://en.wikipedia.org/wiki/Automatic_summarization) to [information extraction](https://en.wikipedia.org/wiki/Information_extraction) and [query answering](https://en.wikipedia.org/wiki/Question_answering) in any given domain; they also cover specific tasks such as detection of adverse drug reactions from social media data and association discovery of diagnosis codes from electronic medical records (EMR).

## Cite

Please cite our papers, code, and dataset if you use them in your work.

deepBioWSD paper, and aforementioned code, and dataset:
```
@article{pesaranghader2019deepbiowsd,
  title={deepBioWSD: effective deep neural word sense disambiguation of biomedical text data},
  author={Pesaranghader, Ahmad and Matwin, Stan and Sokolova, Marina and Pesaranghader, Ali},
  journal={Journal of the American Medical Informatics Association},
  volume={26},
  number={5},
  pages={438--446},
  year={2019},
  publisher={Oxford University Press}
}
```

Single Bidirectional LSTM for WSD:
```
@inproceedings{pesaranghader2018one,
  title={One single deep bidirectional LSTM network for word sense disambiguation of text data},
  author={Pesaranghader, Ahmad and Pesaranghader, Ali and Matwin, Stan and Sokolova, Marina},
  booktitle={Canadian Conference on Artificial Intelligence},
  pages={96--107},
  year={2018},
  organization={Springer}
}
```

<!---_**The project is under further completion at this moment.**_--->

<br/>
<br/>
