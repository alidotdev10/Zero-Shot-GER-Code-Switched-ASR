# Zero-Shot Generative Error Correction (GER) for Code-Switched ASR

## 📖 Project Overview & Academic Motivation
This project is an experimental evaluation of **Generative Error Correction (GER)** applied to Code-Switched Automatic Speech Recognition (CS-ASR). 

It is directly inspired by the 2023 paper *"Generative Error Correction for Code-Switching Speech Recognition Using Large Language Models"* by Chen et al. (Speech and Language Laboratory, Nanyang Technological University). While their research successfully demonstrated that LLMs can drastically reduce Mixed Error Rates (MER) using Low-Rank Adaptation (LoRA), this repository explores a modern extension: **Can the latest generation of open-weights foundation models (LLaMA-3.1) achieve this GER zero-shot, bypassing the need for parameter tuning?**

## 🏗️ Architecture & Methodology
Traditional acoustic models suffer heavily from **Language Dominance Bias** when processing code-switched audio. When a speaker switches between two languages (e.g., English and Urdu/Hindi), baseline models often force the entire sequence into a single dominant script, causing severe cross-lingual phonetic hallucinations.

To test zero-shot recovery against this bias, this pipeline utilizes a cascaded architecture:
1. **Acoustic Tokenization:** `whisper-large-v3` (via Groq) is used to transcribe the raw multilingual acoustic stream.
2. **Generative Error Correction:** `llama-3.1-8b-instant` is deployed as a semantic post-processor. Using a highly constrained, language-agnostic recovery prompt, the LLM is instructed to zero-shot infer language boundaries, identify cross-script phonetic hallucinations, and restore the original code-switched state.

## 📊 Visual Evaluation: Language Dominance Bias vs. Zero-Shot GER
During evaluation on mixed-language datasets, the baseline ASR completely failed to retain English tokens if the latter half of the utterance was in Urdu/Hindi. 

*(Replace the placeholder image below with the screenshot you took of the UI)*
![GER Comparison Dashboard](link_to_your_screenshot_image_here.png)

| Input Audio Characteristics | Baseline (`whisper-large-v3`) | LLM Recovery (`llama-3.1-8b`) |
| :--- | :--- | :--- |
| **Mixed English / Urdu** | Suffers from Language Dominance Bias. English tokens (e.g., "I mean, Turkey") are hallucinated into foreign characters (e.g., "آمی ترکیہ"). | Detects phonetic entrapment and zero-shot recovers the code-switched boundaries perfectly. |

## ⚙️ Installation & Usage
To run this pipeline locally:

1. Clone the repository:
```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
   cd Your-Repo-Name

2. Install the required dependencies:
    pip install -r requirements.txt

3. Set your Groq API key in the app.py script.

4. Launch the Gradio evaluation interface:

    python app.py

Note: The script automatically downloads a small batch of evaluation audio from the MohamedRashad/multilingual-tts Hugging Face dataset.