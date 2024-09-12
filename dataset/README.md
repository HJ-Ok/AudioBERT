## Dataset
### AuditoryBench
AuditoryBench is the first dataset aimed at evaluating language models' auditory knowledge. It comprises:
- **Animal Sound Recognition**: Predict the animal based on an onomatopoeic sound (e.g., "meow").
- **Sound Pitch Comparison**: Compare the pitch of different sound sources.

Animal Sound Recognition
- animal: The name of the animal that the sound corresponds to (e.g., cat).
- description: Description of the animal sound (e.g., meow).
- sentence: A sentence involving the sound, with a [MASK] placeholder for the animal (e.g., "Meow is the sound a [MASK] makes.").

Sound Pitch Comparison
- span1: Description of the first sound (e.g., "sound of a synthesizer").
- span2: Description of the second sound (e.g., "acoustic bass").
- sentence: A sentence comparing the two sounds (e.g., "The sound of a synthesizer typically has a [MASK] pitch than an acoustic bass.").
- answer: The correct comparison (e.g., "higher").

### Data generation code 
you need to download [**LAION-Audio-630K** dataset](https://huggingface.co/datasets/Meranti/CLAP_freesound) for generation


This dataset is built using audio-text pairs from the **LAION-Audio-630K** dataset and includes both training, development, and test sets. Further, we augment the data with audio from Wikipedia for broader generalization.

| Task                  | Train | Dev | Test | Wiki | Total |
|-----------------------|-------|-----|------|------|-------|
| Animal Sound Recognition | 4,211 | 593 | 1,211 | 197 | 6,212 |
| Sound Pitch Comparison  | 8,312 | 1,178 | 2,387 | 3,625 | 15,502 |

![AudioBERT_datapipline_figure2 (4)_page-0001](https://github.com/user-attachments/assets/1d1093e9-c07e-4a81-9ef0-5f2ee860cf5c)
