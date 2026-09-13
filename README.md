# CSL75 Skill Enhancement Laboratory: Generative AI

Lab work for CSL75, a 0:1:1 credit laboratory course covering generative
models, large language models, prompt engineering, word embeddings, Hugging
Face, LangChain, RNNs, LSTMs and transformers.

## Repository layout

| Path | Contents |
| --- | --- |
| `experiment1_multimodal/` | Experiment 1: multimodal generation and cross modal alignment scoring |
| `CSL75_lab_syllabus.txt` | Course contents, the list of experiments and the evaluation scheme |

## Experiments

| No. | Title | Status |
| --- | --- | --- |
| 1 | Multimodal generative AI: text, image and audio from one theme | Implemented |
| 2 | Word embeddings for prompt enrichment | Not started |
| 3 | TensorFlow DNN for academic performance prediction | Not started |
| 4 | Hugging Face summarization of academic reports | Not started |
| 5 | Pretrained word vectors, vector arithmetic and PCA visualization | Not started |
| 6 | Custom Word2Vec on a domain specific corpus | Not started |
| 7 | LangChain with Cohere, document loading and prompt templates | Not started |
| 8 | Hugging Face sentiment analysis pipeline | Not started |
| 9 | LangChain with Pydantic for structured Wikipedia extraction | Not started |
| 10 | Retrieval augmented chatbot over the Indian Penal Code | Not started |
| 11 | PyTorch RNN for time series forecasting | Not started |
| 12 | LSTM text generation | Not started |

## Running the code

Each experiment folder is self contained and carries its own README,
`requirements.txt` and Colab notebook. Start there.

Generated outputs are not committed. They are produced by running the code and,
in the Colab workflow, downloaded as an archive at the end of the notebook.

## Credentials

Nothing in this repository contains an API key. Keys are read from environment
variables at runtime, and the notebooks prompt for them with a masked input so
they are never written into a saved file.
