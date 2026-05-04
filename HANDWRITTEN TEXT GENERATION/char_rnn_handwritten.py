#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # Allows running data-prep without torch installed.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_TRENDING_URL = "https://huggingface.co/papers/trending"
DEFAULT_DATA_PATH = Path("data/hf_trending_corpus.txt")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/char_rnn_hf_trending.pt")
DEFAULT_IMAGE_OUTPUT_PATH = Path("outputs/generated_handwritten.png")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError(
            "PyTorch is not installed. Install dependencies first: pip install -r requirements.txt"
        )


def require_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError(
            "Pillow is not installed. Install dependencies first: pip install -r requirements.txt"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def clean_html_text(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", "", value)
    unescaped = html.unescape(no_tags)
    normalized = re.sub(r"\s+", " ", unescaped)
    return normalized.strip()


def fetch_trending_papers(
    url: str, max_papers: int = 60, timeout: int = 30
) -> List[Dict[str, str]]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    page = response.text

    h3_pattern = re.compile(
        r'<h3[^>]*>\s*<a href="(?P<href>/papers/[^"]+)"[^>]*>(?P<title>.*?)</a>\s*</h3>',
        re.IGNORECASE | re.DOTALL,
    )
    summary_pattern = re.compile(
        r'<p[^>]*class="[^"]*line-clamp-2[^"]*text-sm[^"]*"[^>]*>(?P<summary>.*?)</p>',
        re.IGNORECASE | re.DOTALL,
    )

    papers: List[Dict[str, str]] = []
    seen_titles = set()
    look_ahead_window = 3000

    for h3_match in h3_pattern.finditer(page):
        title = clean_html_text(h3_match.group("title"))
        if not title:
            continue
        dedup_key = title.lower()
        if dedup_key in seen_titles:
            continue

        snippet = page[h3_match.end() : h3_match.end() + look_ahead_window]
        summary_match = summary_pattern.search(snippet)
        summary = clean_html_text(summary_match.group("summary")) if summary_match else ""
        if not summary:
            continue

        href = h3_match.group("href")
        paper_url = f"https://huggingface.co{href}"
        papers.append({"title": title, "summary": summary, "url": paper_url})
        seen_titles.add(dedup_key)

        if len(papers) >= max_papers:
            break

    if not papers:
        raise RuntimeError(
            "No papers could be extracted from the provided page. The HTML structure may have changed."
        )

    return papers


def build_corpus_from_papers(papers: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for item in papers:
        chunks.append(f"Title: {item['title']}\nSummary: {item['summary']}\n")
    return "\n".join(chunks).strip() + "\n"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_or_prepare_corpus(
    data_path: Path,
    url: str,
    max_papers: int,
    timeout: int,
    refresh_data: bool,
) -> Tuple[str, int]:
    if data_path.exists() and not refresh_data:
        corpus = data_path.read_text(encoding="utf-8")
        return corpus, 0

    papers = fetch_trending_papers(url=url, max_papers=max_papers, timeout=timeout)
    corpus = build_corpus_from_papers(papers)
    save_text(data_path, corpus)
    return corpus, len(papers)


class CharSequenceDataset(Dataset):
    def __init__(self, encoded_text: List[int], seq_len: int, step: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2.")
        if step < 1:
            raise ValueError("step must be >= 1.")
        if len(encoded_text) <= seq_len:
            raise ValueError(
                f"Corpus is too small ({len(encoded_text)} chars). Increase data size or lower seq_len."
            )

        self.encoded_text = encoded_text
        self.seq_len = seq_len
        self.starts = list(range(0, len(encoded_text) - seq_len, step))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        x = self.encoded_text[start : start + self.seq_len]
        y = self.encoded_text[start + 1 : start + self.seq_len + 1]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


if nn is not None:
    class CharRNN(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embedding(x)
            out, hidden = self.rnn(emb, hidden)
            logits = self.output(out)
            return logits, hidden
else:
    class CharRNN:  # pragma: no cover
        pass


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(args: argparse.Namespace) -> None:
    require_torch()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    checkpoint_path = Path(args.checkpoint_path)
    corpus, fetched_count = load_or_prepare_corpus(
        data_path=data_path,
        url=args.url,
        max_papers=args.max_papers,
        timeout=args.timeout,
        refresh_data=args.refresh_data,
    )

    unique_chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(unique_chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = [stoi[ch] for ch in corpus]

    dataset = CharSequenceDataset(encoded, seq_len=args.seq_len, step=args.step)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError(
            "No batches were created. Reduce --batch-size or increase corpus size."
        )

    device = pick_device(args.device)
    model = CharRNN(
        vocab_size=len(unique_chars),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Data path: {data_path.resolve()}")
    if fetched_count:
        print(f"Fetched {fetched_count} paper summaries from {args.url}")
    print(f"Corpus length: {len(corpus)} chars | Vocab size: {len(unique_chars)}")
    print(f"Training samples: {len(dataset)} | Batches/epoch: {len(dataloader)}")
    print(f"Device: {device}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step_idx, (x, y) in enumerate(dataloader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            running_loss += loss.item()

            if args.print_every > 0 and step_idx % args.print_every == 0:
                avg = running_loss / args.print_every
                print(f"Epoch {epoch}/{args.epochs} | Step {step_idx} | Loss {avg:.4f}")
                running_loss = 0.0

        if args.print_every <= 0:
            avg = running_loss / max(1, len(dataloader))
            print(f"Epoch {epoch}/{args.epochs} | Loss {avg:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "model_config": {
                "vocab_size": len(unique_chars),
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
            "training_config": {
                "seq_len": args.seq_len,
                "step": args.step,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "source_url": args.url,
                "data_path": str(data_path),
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path.resolve()}")


@torch.no_grad() if torch is not None else (lambda func: func)
def generate_text(
    model,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: str,
    length: int,
    temperature: float,
    device: str,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    model.eval()
    hidden = None

    if not prompt:
        prompt = random.choice(list(stoi.keys()))

    result_chars = list(prompt)

    for ch in prompt:
        idx = torch.tensor([[stoi.get(ch, 0)]], dtype=torch.long, device=device)
        _, hidden = model(idx, hidden)

    current_char = prompt[-1]
    for _ in range(length):
        x = torch.tensor([[stoi.get(current_char, 0)]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=0)
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = itos[sampled_idx]
        result_chars.append(next_char)
        current_char = next_char

    return "".join(result_chars)


def generate(args: argparse.Namespace) -> None:
    require_torch()
    require_pillow()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = pick_device(args.device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ckpt["model_config"]
    model = CharRNN(
        vocab_size=model_cfg["vocab_size"],
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()} if isinstance(next(iter(ckpt["itos"].keys())), str) else ckpt["itos"]

    generated = generate_text(
        model=model,
        stoi=stoi,
        itos=itos,
        prompt=args.prompt,
        length=args.length,
        temperature=args.temperature,
        device=device,
    )

    image_path = Path(args.output_image)
    render_handwritten_image(
        text=generated,
        image_path=image_path,
        width=args.image_width,
        padding=args.image_padding,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        font_path=args.font_path,
    )
    print(f"Saved generated image: {image_path.resolve()}")


def pick_font(font_size: int, font_path: str = ""):
    if font_path:
        custom = Path(font_path)
        if not custom.exists():
            raise FileNotFoundError(f"Font file not found: {custom}")
        return ImageFont.truetype(str(custom), font_size)

    candidates = [
        Path("C:/Windows/Fonts/segoesc.ttf"),  # Segoe Script
        Path("C:/Windows/Fonts/comic.ttf"),    # Comic Sans MS
        Path("C:/Windows/Fonts/calibrii.ttf"), # Calibri Italic
    ]
    for font in candidates:
        if font.exists():
            return ImageFont.truetype(str(font), font_size)

    return ImageFont.load_default()


def wrap_text_for_width(
    draw: "ImageDraw.ImageDraw", text: str, font, max_width: int
) -> str:
    wrapped_lines: List[str] = []
    for paragraph in text.splitlines():
        if paragraph.strip() == "":
            wrapped_lines.append("")
            continue

        words = paragraph.split(" ")
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
            if right - left <= max_width:
                current = candidate
            else:
                wrapped_lines.append(current)
                current = word
        wrapped_lines.append(current)

    return "\n".join(wrapped_lines)


def render_handwritten_image(
    text: str,
    image_path: Path,
    width: int,
    padding: int,
    font_size: int,
    line_spacing: int,
    font_path: str = "",
) -> None:
    if width <= 2 * padding:
        raise ValueError("image_width must be larger than 2 * image_padding.")

    font = pick_font(font_size=font_size, font_path=font_path)

    temp = Image.new("RGB", (width, 200), (248, 244, 236))
    temp_draw = ImageDraw.Draw(temp)
    wrapped = wrap_text_for_width(
        draw=temp_draw,
        text=text,
        font=font,
        max_width=width - 2 * padding,
    )
    left, top, right, bottom = temp_draw.multiline_textbbox(
        (0, 0),
        wrapped,
        font=font,
        spacing=line_spacing,
    )
    text_height = bottom - top
    height = max(300, text_height + 2 * padding)

    image = Image.new("RGB", (width, height), (248, 244, 236))
    draw = ImageDraw.Draw(image)

    for y in range(padding // 2, height, max(font_size + 8, 34)):
        draw.line((padding // 3, y, width - padding // 3, y), fill=(220, 214, 203), width=1)

    draw.multiline_text(
        (padding, padding),
        wrapped,
        font=font,
        fill=(48, 53, 70),
        spacing=line_spacing,
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="PNG")


def prepare_data(args: argparse.Namespace) -> None:
    data_path = Path(args.output)
    papers = fetch_trending_papers(url=args.url, max_papers=args.max_papers, timeout=args.timeout)
    corpus = build_corpus_from_papers(papers)
    save_text(data_path, corpus)
    print(f"Saved corpus to: {data_path.resolve()}")
    print(f"Papers extracted: {len(papers)}")
    print(f"Corpus length: {len(corpus)} characters")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Character-level RNN for handwritten-like text generation using Hugging Face trending papers text."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-data", help="Fetch and build corpus from the Hugging Face trending papers page.")
    prep.add_argument("--url", default=DEFAULT_TRENDING_URL, help="Source URL for paper text.")
    prep.add_argument("--output", default=str(DEFAULT_DATA_PATH), help="Output corpus text file.")
    prep.add_argument("--max-papers", type=int, default=60, help="Maximum number of unique papers to scrape.")
    prep.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    prep.set_defaults(func=prepare_data)

    train_parser = subparsers.add_parser("train", help="Train a character-level LSTM.")
    train_parser.add_argument("--url", default=DEFAULT_TRENDING_URL, help="Source URL for paper text.")
    train_parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to corpus text file.")
    train_parser.add_argument("--refresh-data", action="store_true", help="Re-fetch corpus even if data file exists.")
    train_parser.add_argument("--max-papers", type=int, default=60, help="Maximum number of unique papers to scrape.")
    train_parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    train_parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Checkpoint output path.")
    train_parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    train_parser.add_argument("--seq-len", type=int, default=120, help="Sequence length.")
    train_parser.add_argument("--step", type=int, default=1, help="Sliding window step.")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    train_parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension.")
    train_parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden size.")
    train_parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout between recurrent layers.")
    train_parser.add_argument("--learning-rate", type=float, default=0.003, help="Learning rate.")
    train_parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping value. 0 disables clipping.")
    train_parser.add_argument("--print-every", type=int, default=50, help="Print loss every N steps. <=0 prints per epoch.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    train_parser.set_defaults(func=train)

    gen = subparsers.add_parser("generate", help="Generate text from a trained checkpoint.")
    gen.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Model checkpoint path.")
    gen.add_argument("--prompt", default="Title: ", help="Initial prompt string.")
    gen.add_argument("--length", type=int, default=500, help="Number of characters to generate.")
    gen.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (>0).")
    gen.add_argument("--output-image", default=str(DEFAULT_IMAGE_OUTPUT_PATH), help="Single output image path.")
    gen.add_argument("--image-width", type=int, default=1400, help="Output image width in pixels.")
    gen.add_argument("--image-padding", type=int, default=70, help="Padding around text in pixels.")
    gen.add_argument("--font-size", type=int, default=42, help="Font size for rendered text.")
    gen.add_argument("--line-spacing", type=int, default=22, help="Line spacing for multiline text.")
    gen.add_argument(
        "--font-path",
        default="",
        help="Optional .ttf font path. Leave empty for auto handwriting-style font fallback.",
    )
    gen.add_argument("--seed", type=int, default=42, help="Random seed.")
    gen.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Generation device.")
    gen.set_defaults(func=generate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
