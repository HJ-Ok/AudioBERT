# AudioBERT 📢 : Audio Knowledge Augmented Language Model

 [**🤗 Dataset**](https://huggingface.co/datasets/HJOK/AuditoryBench) | [**📄 arXiv**](https://arxiv.org/abs/2409.08199) 

This repository contains the [model code](model/) and the [dataset](dataset/) of our AudioBERT and AuditoryBench.

## Updates
- **(2025.01.13)**: Our paper has been accepted at ICASSP 2025! We also added multiple-choice options in AuditoryBench.
- **(2024.09.26)**: Gaudi HPU training code added. We updated the Auditory Knowledge Span Detector and CLAP retrieval code. (There are some issues that will be resolved)
  - [ ] Solve the unstable issue in AudioBERT LoRA training
  - [ ] Code refactoring for easy to use

## Introduction
Language models like BERT, while powerful in text-based tasks, often lack auditory knowledge. This project introduces **AudioBERT**, a method to inject auditory knowledge into language models via a retrieval-based approach, improving performance on auditory knowledge tasks.  
To evaluate this, we introduce **AuditoryBench**, a dataset featuring tasks like animal sound recognition and sound pitch comparison. AudioBERT leverages **CLAP** (Contrastive Language-Audio Pretraining) for effective audio-text matching.

<p align="center">
    <img src="https://github.com/user-attachments/assets/a2093991-fc1c-4d0a-9dca-fa3aa284741c" alt="AudioBERT" style="width: 30%; height: auto;">
</p>



## Dataset
### AuditoryBench
You can also see our dataset in huggingface and download it by following code.  

```Python
from datasets import load_dataset

animal_sound_recognition_dataset = load_dataset("HJOK/AuditoryBench","animal_sound_recognition")
sound_pitch_comparsion_dataset = load_dataset("HJOK/AuditoryBench","sound_pitch_comparsion")
```

AuditoryBench is the first dataset aimed at evaluating language models' auditory knowledge. It comprises:
- **Animal Sound Recognition**: Predict the animal based on an onomatopoeic sound (e.g., "meow").
- **Sound Pitch Comparison**: Compare the pitch of different sound sources.

This dataset is built using audio-text pairs from the **LAION-Audio-630K** dataset and includes both training, development, and test sets. Further, we augment the data with audio from Wikipedia for broader generalization.
You can download our dataset and look detailed dataset generation process [here](dataset/README.md).

| Task                  | Train | Dev | Test | Wiki | Total |
|-----------------------|-------|-----|------|------|-------|
| Animal Sound Recognition | 4,211 | 593 | 1,211 | 197 | 6,212 |
| Sound Pitch Comparison  | 8,312 | 1,178 | 2,387 | 3,625 | 15,502 |

![AudioBERT_datapipline_figure2 (4)_page-0001](https://github.com/user-attachments/assets/0551fd69-4ad4-4f22-b106-0b9959c2d930)



## Model
### AudioBERT
AudioBERT uses a retrieval-based framework to inject auditory knowledge into language models. Its key components include:
- **Auditory Knowledge Span Detector**: This component detects text spans where auditory knowledge is needed, identifying key tokens related to sounds or objects for audio retrieval.
- **CLAP Retrieval**: Once the span is identified, CLAP retrieves the most relevant audio by matching the text span with audio samples. This embedding is then added to the model to enhance auditory understanding.
- **AudioBERT (LoRA)**: Dynamically adapts the model with auditory embeddings when necessary, ensuring general performance on other language tasks.

Detailed codes are available [here](model/README.md).

![AudioBERT_model (1)_page-0001](https://github.com/user-attachments/assets/0ec1c8d3-4f18-4278-b9b0-2cf4d941263e)



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

## Installation
(TBD)  
To install and run AudioBERT, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/HJ-Ok/AudioBERT.git
    cd AudioBERT
    ```

## License
(TBD)  
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
  author={Ok, Hyunjong and Yoo, Suho and Lee, Jaeho},
  journal={arXiv preprint arXiv:2409.08199},
  year={2024}
}
```
