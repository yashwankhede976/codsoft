# Character-Level RNN for Handwritten-Like Text Generation

This project trains a character-level LSTM (RNN) on text collected from:

- https://huggingface.co/papers/trending

It then generates new character sequences that mimic learned writing patterns.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Build the dataset corpus

```bash
python char_rnn_handwritten.py prepare-data --output data/hf_trending_corpus.txt --max-papers 60
```

This creates a text corpus from paper titles and summaries on the trending page.

## 3) Train the character-level RNN

```bash
python char_rnn_handwritten.py train --data-path data/hf_trending_corpus.txt --epochs 40 --seq-len 120 --batch-size 64 --hidden-size 256 --num-layers 2
```

Checkpoint is saved by default to:

- `checkpoints/char_rnn_hf_trending.pt`

## 4) Generate new text

```bash
python char_rnn_handwritten.py generate --checkpoint-path checkpoints/char_rnn_hf_trending.pt --prompt "Title: " --length 500 --temperature 0.8 --output-image outputs/generated_handwritten.png
```

This saves the output as a single image file:

- `outputs/generated_handwritten.png`

## Notes

- The Hugging Face page content changes over time, so retraining with `--refresh-data` updates the corpus.
- This is character-level generation, so output may include creative spelling and structure variation.
