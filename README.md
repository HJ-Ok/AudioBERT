# AudioBERT 📢 : Audio Knowledge Augmented Language Model
This repository contains the [model code](model/README.md) and the [dataset](dataset/README.md) of our AudioBERT and AuditoryBench.
Now datasets and generation prompts are available (The detailed and refactored code will be updated after the ICASSP 2025 review.)

## Introduction
Language models like BERT, while powerful in text-based tasks, often lack auditory knowledge. This project introduces **AudioBERT**, a method to inject auditory knowledge into language models via a retrieval-based approach, improving performance on auditory knowledge tasks.  
To evaluate this, we introduce **AuditoryBench**, a dataset featuring tasks like animal sound recognition and sound pitch comparison. AudioBERT leverages **CLAP** (Contrastive Language-Audio Pretraining) for effective audio-text matching.

<p align="center">
    <img src="https://github.com/user-attachments/assets/4e2c9d61-cdf0-41d4-a64d-5e9c9121b2a6" alt="AudioBERT" style="width: 30%; height: auto;">
</p>


## Dataset
### AuditoryBench
AuditoryBench is the first dataset aimed at evaluating language models' auditory knowledge. It comprises:
- **Animal Sound Recognition**: Predict the animal based on an onomatopoeic sound (e.g., "meow").
- **Sound Pitch Comparison**: Compare the pitch of different sound sources.

This dataset is built using audio-text pairs from the **LAION-Audio-630K** dataset and includes both training, development, and test sets. Further, we augment the data with audio from Wikipedia for broader generalization.
You can download our dataset and look detailed dataset generation process [here](dataset/README.md).

| Task                  | Train | Dev | Test | Wiki | Total |
|-----------------------|-------|-----|------|------|-------|
| Animal Sound Recognition | 4,211 | 593 | 1,211 | 197 | 6,212 |
| Sound Pitch Comparison  | 8,312 | 1,178 | 2,387 | 3,625 | 15,502 |

![AudioBERT_datapipline_figure2 (4)_page-0001](https://github.com/user-attachments/assets/1d1093e9-c07e-4a81-9ef0-5f2ee860cf5c)


## Model
### AudioBERT
AudioBERT uses a retrieval-based framework to inject auditory knowledge into language models. Its key components include:
- **Auditory Knowledge Span Detector**: Identifies relevant auditory spans in text.
- **CLAP**: Retrieves relevant audio embeddings from text spans.
- **LoRA (Low-Rank Adaptation)**: Dynamically adapts the model with auditory embeddings when necessary, ensuring general performance on other language tasks.

Detailed codes are available [here](model/README.md).

![AudioBERT_model (1)_page-0001](https://github.com/user-attachments/assets/e026332d-faf5-4261-bbfe-6062d8c7de0a)


### Training
We employ a BERT-base model for the auditory knowledge spandetector. We trained with 5 epochs with a batch size of 16, a learning rate of 1×10−5, and utilizing AdamW optimizer.

We experimented using BERT for the language model and employed an AST encoder for auditory knowledge embedding injecting. We trained with 20 epochs with a batch size of 32, a learning rate of 3 × 10−4, and utilizing AdamW optimizer. For LoRA, we set the rank and alpha to 64 and 128.

## Results
AudioBERT outperforms baseline models such as BERT, RoBERTa, Gemma2-2B, and LlaMA3.1-8B in auditory tasks, achieving significantly higher accuracy on both AuditoryBench tasks in the test set.

| Model           | Animal Sound (Acc) | Sound Pitch (Acc) | Combined (Acc) |
|-----------------|--------------------|-------------------|----------------|
| BERT-large      | 15.85              | 58.90             | 44.41          |
| RoBERTa-large   | 14.70              | 56.64             | 42.52          |
| Gemma2-2B       | 15.11              | 60.45             | 45.19          |
| LLaMA3.1-8B     | 21.80              | 62.55             | 48.83          |
| **AudioBERT**   | **36.69**          | **76.31**         | **62.97**      |

(TBD)
## Installation
To install and run AudioBERT, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/HJ-Ok/AudioBERT.git
    cd AudioBERT
    ```

## License
```
MIT license

Copyright (c) 2024 Hyunjong Ok

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Citation
If you use this code, please cite the following paper:
```
@article{ok2024audiobert,
  title={AudioBERT: Audio Knowledge Augmented Language Model},
  author={Hyunjong Ok and Suho Yoo and Jaeho Lee},
  journal={arXiv preprint arXiv:2409.xxxxx},
  year={2024}
}
```
