# deepBioWSD: A One-size-fits-all Deep Bidirectional LSTM for Word Sense Disambiguation of Biomedical Text Data

## Background
Word sense ambiguity is a pervasive characteristic of natural language. For example, the word _cold_ has several senses and may refer to a disease, a temperature sensation, or an environmental condition. The specific sense intended is determined by the textual context in which an instance of the ambiguous word appears. In **"I am taking aspirin for my cold"** the _disease_ sense is intended, in **"Let's go inside, I'm cold"** the _temperature sensation_ sense is meant, while **"It's cold today, only 2 degrees"**, implies the _environmental condition_ sense. Therefore, automatically identifying the intended sense of ambiguous words improves the proper inference of biomedical textual data for clinical and biomedical applications. 

## deepBioWSD Network
This project, after laying out and developing a novel deep Bidirectional Long Short-Term Memory (BLSTM) learning model, attempts to address this crucial problem in NLP. We evaluate the effectiveness of such model over the task of word sense disambiguation (WSD) in the biomedical domain. In brief, the designed network, after being initialized by pre-trained concept vectors (i.e. concept embeddings), would be trained on the biomedical textual data fed to the model. Finally, we test the model on a list of held-out data. As to calculation of pre-trained _concept embeddings_, we make use of UMLS and MEDLINE abstracts and also employ Pointwise Mutual Information (PMI) and Latent Semantic Analysis (LSA). The experimental result on the MSH-WSD dataset shows the developed deep learning model outperforms other common measures in the field in terms of accuracy. 

## Project Outcome
The outcome of this project is directly applicable to a wide range of NLP applications. These applications run the gamut from machine translation as well as text summarization to information extraction and query answering in any given domain; they also cover specific tasks such as detection of adverse drug reactions from social media data and association discovery of diagnosis codes from electronic medical records (EMR).


_**The project is under completion at this moment.**_

<br/>
<br/>

<sub>IWERA © 2017</sub>
