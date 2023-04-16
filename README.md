# Weak supervision for sentiment analisys (WSQASA)

- Victor A. K. Tomita (ICMC/USP) | akihito012@usp.br (corresponding author)
- Ricardo M. Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

## Astract
---

The growth of social networks, e-commerce, and journalistic media has resulted in the proliferation of opinions on various topics. Companies and government agencies are interested in understanding their customers' opinions about their products and services. Automatic sentiment analysis methods can be used to extract the general sentiment about a product. However, traditional sentiment analysis methods are inflexible in dealing with human queries, which tend to ask questions. Therefore, question-and-answer (QA) systems for sentiment analysis offer a promising alternative. This paper proposes a new method called Weak Supervision for Question and Answering Sentiment Analysis (WSQASA) that fine-tunes and extracts sentiment through QA models in an unsupervised manner. We investigate question-generation models associated with sentiment filters for weak supervision, generating domain-specific question-and-answer pairs for fine-tuning the QA model. Our method enables the generation of domain-specific question-and-answer pairs for fine-tuning the QA model, which significantly enhances the QA-based sentiment analysis results, even without the usage of labeled data.

## WSQASA - pipeline
![Proposal](/images/WSQASA_pipe.png)

## Results
![Proposal](/images/f1_comp.png)