## Model
### AudioBERT
AudioBERT uses a retrieval-based framework to inject auditory knowledge into language models. Its key components include:
- **Auditory Knowledge Span Detector**: This component detects text spans where auditory knowledge is needed, identifying key tokens related to sounds or objects for audio retrieval.
- **CLAP Retrieval**: Once the span is identified, CLAP retrieves the most relevant audio by matching the text span with audio samples. This embedding is then added to the model to enhance auditory understanding.
- **AudioBERT (LoRA)**: Dynamically adapts the model with auditory embeddings when necessary, ensuring general performance on other language tasks.

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


## Usage
To run AudioBERT on a sample auditory knowledge task, use the following commands:
(TBD)

1. **Training the Model**:
    ```bash
    python train.py
    ```

2. **Evaluating the Model**:
    ```bash
    python evaluate.py
    ```
