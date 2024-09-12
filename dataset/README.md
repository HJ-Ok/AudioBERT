## Dataset
### AuditoryBench
AuditoryBench is the first dataset aimed at evaluating language models' auditory knowledge. It comprises:
- **Animal Sound Recognition**: Predict the animal based on an onomatopoeic sound (e.g., "meow").
- **Sound Pitch Comparison**: Compare the pitch of different sound sources.

### (Description TBD) 
Animal Sound Recognition
- animal:
- description:
- sentence:

Sound Pitch Comparison
- span1:
- span2:
- sentence:
- answer:

### Data generation code 
you need to download [**LAION-Audio-630K** dataset](https://huggingface.co/datasets/Meranti/CLAP_freesound) for generation


This dataset is built using audio-text pairs from the **LAION-Audio-630K** dataset and includes both training, development, and test sets. Further, we augment the data with audio from Wikipedia for broader generalization.

| Task                  | Train | Dev | Test | Wiki | Total |
|-----------------------|-------|-----|------|------|-------|
| Animal Sound Recognition | 4,211 | 593 | 1,211 | 197 | 6,212 |
| Sound Pitch Comparison  | 8,312 | 1,178 | 2,387 | 3,625 | 15,502 |

![AudioBERT_datapipline_figure2 (4)_page-0001](https://github.com/user-attachments/assets/1d1093e9-c07e-4a81-9ef0-5f2ee860cf5c)