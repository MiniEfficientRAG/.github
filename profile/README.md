# EfficientRAG: Multi-Hop Question Answering with Efficient Retrieval

This repository contains a **reproduction and adaptation** of the EfficientRAG framework, designed to perform multi-hop question answering efficiently by replacing expensive LLM calls during retrieval with lightweight, trained models.

The implementation is adapted to run on **Google Colab** (using a single A100 GPU with 80GB memory) and utilizes a local **Meta-Llama-3-8B-Instruct** server via [`vllm`](https://docs.vllm.ai/en/latest/), as the generator, replacing the proprietary OpenAI models used in the original paper.

## 📊 Project Poster

![Project Poster](https://github.com/Mini-EfficientRAG/Materials/blob/main/Final_Poster.jpg)
*([Click here to download the high-resolution PDF](https://github.com/Mini-EfficientRAG/Materials/blob/main/Final_Poster.pdf))*

## 📄 Project Overview

EfficientRAG addresses the challenge of multi-hop question answering—where answering a question requires gathering evidence from multiple documents sequentially—without the high latency and cost of "Iterative RAG" (which uses an LLM at every step).

Instead, this project implements two trainable components:
1.  **Labeler (Token Classification):** Identifies essential tokens (keywords) in a passage that are relevant to the query and determines if a retrieved passage is relevant or irrelevant, discarding noise.
2.  **Filter (Binary Classification):** Determines the next query (the clearer query) based on the kept chunks and the previous query.

By chaining these components, the system can iteratively retrieve and filter information before passing a compressed, high-quality context to the final LLM for answer generation.

## 📁 Repository Structure

The project is divided into three phases, each contained in a self-sufficient Jupyter Notebook:

```text
EfficientRAG/
│
├── EfficientRAG.ipynb           # PHASE 1: Data Synthesis
│   ├── Sets up the local vLLM server (Llama-3).
│   ├── Downloads the HotpotQA dataset.
│   ├── Runs the data synthesis pipeline: Query Decomposition, Token Labeling, and Negative Sampling.
│   └── Dynamically patches source code for local execution.
│
├── EfficientRAG_Training.ipynb  # PHASE 2: Model Training
│   ├── Loads the synthetic data generated in Phase 1.
│   ├── Trains the Filter Model (DeBERTa-v3-large).
│   ├── Trains the Labeler Model (DeBERTa-v3-large).
│   └── Exports the trained model checkpoints.
│
├── EfficientRAG_Inference.ipynb # PHASE 3: Inference & Evaluation
│   ├── Loads the trained Filter and Labeler models.
│   ├── Executes the full inference pipeline: Retrieval -> Filtering -> Labeling -> Generation.
│   └── Evaluates performance (EM/F1) on the HotpotQA dataset.
│
├── Report.pdf                       # Detailed project report with methodology and results.
├── Final_Poster.pdf                 # High-res project poster.
├── Final_Poster.png               # Poster preview image.
└── README.md                        # This file.
````

## 🛠️ Prerequisites & Dependencies

The project is designed to run on **Google Colab** with an A100 GPU (or similar environment with \>80GB VRAM).

The notebooks automatically install the following key libraries, others are installed based on the `requirements.txt` file of the original repo:

  * `vllm` (for local LLM serving)
  * `transformers>=4.45.2`
  * `accelerate>=1.1.0`
  * `peft>=0.16.0`
  * `faiss-cpu` (or `faiss-gpu`)
  * `wandb` (for experiment tracking)
  * `rich`, `tenacity`, `datasets`

## ⚠️ Important: vLLM Server & Colab Stability

**Known Issue:** When running `vllm` in Colab, processing large batches of samples continuously (typically \>200) can cause the KV cache to saturate, leading to `RequestTimedOutError` or process hangs.

**Solution:**
The code includes logic to handle this, but if you encounter a timeout:

1.  **Terminate the vLLM server:** Execute `proc.terminate()` in the relevant cell or manually stop the cell.
2.  **Restart the Server:** Re-run the cell responsible for deploying the Llama model (`subprocess.Popen(...)`).

*Note:* Several data synthesis loops in `EfficientRAG.ipynb` are designed to automatically restart the server periodically to mitigate this.

## 🚀 How to Run

1.  **Clone/Download:** Clone this repository or download the notebooks.
2.  **Upload to Colab:** Upload the notebooks to Google Colab.
3.  **Phase 1 (Data Synthesis):**
      * Open [`EfficientRAG.ipynb`](https://github.com/Mini-EfficientRAG/Materials/blob/main/src/EfficientRAG.ipynb).
      * Run all cells. You will need a **Hugging Face Token** with access to `Meta-Llama-3-8B-Instruct`.
      * This will generate the training data (`.jsonl` files).
4.  **Phase 2 (Training):**
      * Open [`EfficientRAG_Training.ipynb`](https://github.com/Mini-EfficientRAG/Materials/blob/main/src/EfficientRAG_Inference.ipynb).
      * Ensure the data from Phase 1 is available (or use the download cell provided in the notebook to fetch pre-computed data).
      * Run the training cells to produce `filter_model.zip` and `labeler_model.zip`.
5.  **Phase 3 (Inference):**
      * Open [`EfficientRAG_Inference.ipynb`](https://github.com/Mini-EfficientRAG/Materials/blob/main/src/EfficientRAG_Inference.ipynb).
      * Upload the trained model zip files.
      * Run the inference pipeline to generate answers and view evaluation metrics.
  
*Note:* I've uploaded my trained models' checkpoints on my Google Drive and they will be automatically downloaded if you run the inference code, but if you want the results on your own models, simply replace the links with yours and do the same.

## 📊 Results

Detailed in the [`report.pdf`](https://github.com/Mini-EfficientRAG/Materials/blob/main/Report.pdf).

## 📝 Acknowledgments

  * Original Paper: [EfficientRAG: Efficient Retriever for Multi-Hop Question Answering](https://aclanthology.org/2024.emnlp-main.199.pdf)
  * Original Repository: [nil-zhuang/efficientrag-official](https://github.com/nil-zhuang/efficientrag-official)
  * Models: [Meta Llama 3](https://llama.meta.com/), [DeBERTa](https://huggingface.co/microsoft/deberta-v3-large), [Contriever](https://huggingface.co/facebook/contriever)

-----

*This project was developed for educational purposes to demonstrate reproducible ML research in resource-constrained environments.*
