# 🦙 Fine-Tuning LLaMA 3 with Unsloth (Colab + Hugging Face)

This project demonstrates how to fine-tune Meta's LLaMA 3 (8B Instruct) model using [Unsloth](https://github.com/unslothai/unsloth) in Google Colab, and push the merged model to [Hugging Face Hub](https://huggingface.co).

---

## 📄 Description

Fine-tuned Meta’s LLaMA 3 (8B Instruct) model using a small subset of the Guanaco dataset consisting of 500 instruction-response pairs. The Guanaco dataset is designed to train language models for instruction-following tasks, making it ideal for conversational or chat-based AI use cases. The fine-tuning was performed using the Unsloth library, which enables fast and memory-efficient training with 4-bit quantized models on limited hardware (like Google Colab free tier). Applied Low-Rank Adaptation (LoRA) to train only a small portion of the model weights, significantly reducing computational cost. While our training dataset was intentionally small for speed and simplicity, larger and more diverse datasets would yield higher-quality, more generalizable responses.

The model used is the `unsloth/llama-3-8b-Instruct-bnb-4bit` — a 4-bit quantized version that runs efficiently even on Colab's free tier (T4 GPU).

Used **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning. Only ~0.5% of model weights were updated. This reduces memory usage and training time significantly.

> **Note:** Fine-tuning on a larger dataset would lead to more coherent and useful responses. This project is optimized for learning and resource constraints.

---

## 📦 Libraries Used

| Library            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `unsloth`          | A high-performance wrapper for fine-tuning LLMs efficiently in Colab       |
| `transformers`     | Hugging Face's core library for working with LLMs and tokenizers           |
| `trl`              | Used for `SFTTrainer`, a supervised fine-tuning trainer for instruction tasks |
| `peft`             | Enables Parameter-Efficient Fine-Tuning via LoRA                           |
| `datasets`         | Loads and manages datasets (like Guanaco)                                  |
| `bitsandbytes`     | Supports 4-bit and 8-bit quantization for efficient model loading          |
| `xformers`         | Optimized GPU kernels for attention layers                                  |
| `accelerate`       | Manages device placement and mixed precision training                      |
| `sentencepiece`    | Tokenization library used by LLaMA and other large models                  |
| `protobuf`         | Used internally by many ML libraries; pinned version avoids conflicts      |

---

## 🔁 Step-by-Step Process

### 📊 Diagram


```
 ┌───────────────────────┐
 │  Load Model           │ ◇── LLaMA 3 4bit via Unsloth
 └──────┬────────────────┘
        │
        ▼
 ┌───────────────────────┐
 │ Attach LoRA           │ ◇── PEFT adapter layers
 └──────┬────────────────┘
        │
        ▼
 ┌───────────────────────┐
 │ Load Dataset          │ ◇── Guanaco (500 samples)
 └──────┬────────────────┘
        │
        ▼
 ┌───────────────────────┐
 │ Fine-Tune             │ ◇── SFTTrainer
 └──────┬────────────────┘
        │
        ▼
 ┌───────────────────────┐
 │ Inference             │ ◇── Streamed predictions
 └──────┬────────────────┘
        │
        ▼
 ┌───────────────────────┐
 │ Merge & Push          │ ◇── Hugging Face Hub (merged 16bit)
 └───────────────────────┘
```



---

## 🧠 Example Prompt

```python
messages = [{"from": "human", "value": "What is recursion?"}]
```

Model responds with a natural-language explanation of recursion.

---

## 📬 Model Link

👉 https://huggingface.co/Sirisha4/llama3-finetuned

---

## ✅ Conclusion

This project demonstrates that you can fine-tune a cutting-edge large language model like LLaMA 3 using only free-tier resources with the right tools. 

By combining Unsloth, LoRA, and 4-bit quantization, we make efficient LLM training approachable for students, researchers, and indie developers.

Whether you're experimenting with chatbots, exploring instructional data, or learning model training workflows — this is a great foundation.

> Want better results? Try using 5k or 50k samples instead of 500, and fine-tune for more epochs.

Happy tuning! 🧠✨
