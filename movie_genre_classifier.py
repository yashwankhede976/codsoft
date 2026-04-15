"""
Movie Genre Classification — TF-IDF + Multiple Classifiers
Dataset: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

Expected files (place in same directory):
  train_data.txt  — format: ID ::: TITLE ::: GENRE ::: PLOT
  test_data.txt   — format: ID ::: TITLE ::: PLOT
  test_data_solution.txt — format: ID ::: TITLE ::: GENRE ::: PLOT

Run:
  python movie_genre_classifier.py
"""

import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
GENRES = [
    "action", "adventure", "animation", "biography", "comedy",
    "crime", "documentary", "drama", "fantasy", "horror",
    "musical", "mystery", "romance", "sci-fi", "thriller"
]

GENRE_KEYWORDS = {
    "action":      ["fight", "explosion", "battle", "hero", "mission", "agent", "war", "attack", "weapon", "pursuit"],
    "adventure":   ["journey", "quest", "explore", "discover", "treasure", "expedition", "wilderness", "map", "legendary", "ancient"],
    "animation":   ["animated", "cartoon", "magical creature", "talking", "fairy", "pixar", "dreamworks", "adventure", "colorful", "whimsical"],
    "biography":   ["life story", "based on", "real events", "historical", "famous", "legacy", "politician", "scientist", "musician", "artist"],
    "comedy":      ["hilarious", "funny", "laugh", "joke", "humor", "awkward", "mishap", "absurd", "silly", "comic"],
    "crime":       ["detective", "murder", "criminal", "heist", "robbery", "police", "investigation", "gangster", "corrupt", "mob"],
    "documentary": ["documentary", "real", "footage", "interview", "world", "culture", "history", "society", "environment", "truth"],
    "drama":       ["emotional", "relationship", "family", "struggle", "loss", "redemption", "conflict", "love", "grief", "passion"],
    "fantasy":     ["magic", "dragon", "wizard", "kingdom", "spell", "enchanted", "mythical", "prophecy", "realm", "elf"],
    "horror":      ["terror", "ghost", "haunted", "monster", "fear", "nightmare", "demon", "supernatural", "cursed", "evil"],
    "musical":     ["singing", "dancing", "song", "music", "broadway", "perform", "stage", "melody", "choreography", "rhythm"],
    "mystery":     ["mystery", "clue", "suspect", "detective", "secret", "puzzle", "unknown", "hidden", "reveal", "conspiracy"],
    "romance":     ["love", "relationship", "romantic", "heart", "passion", "couple", "affection", "attraction", "soulmate", "marry"],
    "sci-fi":      ["space", "alien", "future", "robot", "technology", "planet", "galaxy", "time travel", "AI", "dystopia"],
    "thriller":    ["suspense", "chase", "danger", "conspiracy", "thriller", "kidnap", "assassin", "spy", "tension", "threat"],
}

def generate_synthetic_dataset(n_per_genre=200, seed=42):
    """Generate a representative synthetic dataset mirroring IMDB plot summaries."""
    rng = np.random.default_rng(seed)
    templates = {
        "action":      "A {adj} hero must {verb} against {enemy} to save {place}. Full of {noun} and high-octane {noun2}.",
        "adventure":   "A brave {adj} explorer embarks on a {noun} to {verb} the {noun2} of {place}.",
        "animation":   "In a magical world, a {adj} {animal} learns to {verb} and discovers the power of {noun}.",
        "biography":   "The inspiring true story of {person}, whose {adj} {noun} changed {place} forever.",
        "comedy":      "A {adj} {person} accidentally {verb} into chaos at {place}, leading to {adj2} situations.",
        "crime":       "A {adj} detective must {verb} a {noun} ring operating in {place} before time runs out.",
        "documentary": "An eye-opening look at {noun} in {place}, featuring {adj} interviews and rare {noun2}.",
        "drama":       "After {noun}, a {adj} family struggles to {verb} while navigating {adj2} relationships in {place}.",
        "fantasy":     "In the {adj} land of {place}, a young {person} must {verb} an ancient {noun} to save the realm.",
        "horror":      "When {person} moves to {place}, they discover a {adj} {noun} that threatens to {verb} them all.",
        "musical":     "A {adj} {person} dreams of {verb} on Broadway and falls in love while rehearsing for {noun}.",
        "mystery":     "After a {adj} disappearance in {place}, detective {person} unravels a {noun} full of dark secrets.",
        "romance":     "Two {adj} strangers meet in {place} and discover {noun} when they least expect it.",
        "sci-fi":      "In {year}, a {adj} scientist discovers that {noun} threatens the entire {place}.",
        "thriller":    "A {adj} {person} uncovers a {noun} conspiracy in {place} and must {verb} to survive.",
    }
    fillers = {
        "adj":    ["determined", "brilliant", "unlikely", "young", "seasoned", "reckless", "mysterious"],
        "adj2":   ["unexpected", "complicated", "hilarious", "dangerous", "heartwarming"],
        "verb":   ["fight", "uncover", "escape", "protect", "discover", "survive", "overcome"],
        "noun":   ["mission", "secret", "truth", "legacy", "danger", "adventure", "power"],
        "noun2":  ["action", "drama", "comedy", "suspense", "beauty", "history"],
        "enemy":  ["terrorists", "corrupt officials", "a rival clan", "an alien force"],
        "place":  ["New York", "ancient Rome", "a small town", "outer space", "the jungle"],
        "person": ["a detective", "a young woman", "an unlikely hero", "a scientist"],
        "animal": ["fox", "bear", "robot", "dragon", "penguin"],
        "year":   ["2157", "2099", "2220", "2045"],
    }
    rows = []
    for genre in GENRES:
        tmpl = templates[genre]
        kws  = GENRE_KEYWORDS[genre]
        for i in range(n_per_genre):
            text = tmpl
            for k, v in fillers.items():
                text = text.replace("{" + k + "}", rng.choice(v))
            # Sprinkle genre-specific keywords to add signal
            n_kw = rng.integers(2, 6)
            chosen_kws = rng.choice(kws, size=n_kw, replace=True)
            text += " " + " ".join(chosen_kws) + "."
            rows.append({"genre": genre, "plot": text})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def load_imdb_file(path: Path) -> pd.DataFrame:
    """Parse the IMDB genre dataset format: ID ::: TITLE ::: GENRE ::: PLOT"""
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                records.append({"genre": parts[2].strip().lower(), "plot": parts[3].strip()})
            elif len(parts) == 3:  # test file without genre
                records.append({"genre": "unknown", "plot": parts[2].strip()})
    return pd.DataFrame(records)


def load_data():
    train_path = Path("train_data.txt")
    if train_path.exists():
        print("✅  Found train_data.txt — loading real IMDB data …")
        df = load_imdb_file(train_path)
    else:
        print("⚠️   train_data.txt not found — using synthetic data for demonstration.")
        print("    Download from: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb\n")
        df = generate_synthetic_dataset(n_per_genre=200)
    return df


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "it","its","this","that","these","those","i","me","my","we","our","you",
    "your","he","his","she","her","they","them","their","who","which","what",
    "when","where","why","how","not","no","nor","so","yet","both","either",
    "also","just","about","after","before","during","while","as","if","then",
    "than","more","most","very","too","much","many","some","any","all","each",
    "every","other","such","one","two","three","new","old","up","out","into",
    "over","through","between","own","same","only","first","last","long",
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)      # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# 3. PIPELINE DEFINITIONS
# ─────────────────────────────────────────────
def build_pipelines():
    tfidf_common = dict(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_common)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_common)),
            ("clf",   LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", n_jobs=-1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_common)),
            ("clf",   CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))),
        ]),
    }


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def evaluate_models(X, y, label_classes):
    pipelines = build_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n" + "="*60)
    print("  5-FOLD CROSS-VALIDATION RESULTS")
    print("="*60)

    for name, pipe in pipelines.items():
        t0 = time.time()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {
            "pipe":    pipe,
            "mean_f1": scores.mean(),
            "std_f1":  scores.std(),
            "time":    elapsed,
            "scores":  scores,
        }
        print(f"\n  {name}")
        print(f"    F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"    Time:          {elapsed:.1f}s")

    # Pick best model and train on full data
    best_name = max(results, key=lambda n: results[n]["mean_f1"])
    print(f"\n  🏆  Best model: {best_name} (F1={results[best_name]['mean_f1']:.4f})")
    best_pipe = results[best_name]["pipe"]
    best_pipe.fit(X, y)

    return results, best_name, best_pipe


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
COLORS = {
    "Naive Bayes":         "#4C72B0",
    "Logistic Regression": "#55A868",
    "Linear SVM":          "#C44E52",
}

def plot_results(df, results, best_name, best_pipe, X, y, label_classes, out_path="genre_classifier_results.png"):
    fig = plt.figure(figsize=(22, 20), facecolor="#0F1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.06)

    title_kw  = dict(color="white", fontsize=11, fontweight="bold", pad=10)
    label_kw  = dict(color="#AAAAAA", fontsize=9)
    tick_kw   = dict(color="#AAAAAA", labelsize=8)

    # ── 1. Genre distribution ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["genre"].value_counts()
    bars = ax1.barh(counts.index, counts.values, color="#4C72B0", edgecolor="#222")
    ax1.set_facecolor("#1A1D27")
    ax1.set_title("Genre Distribution", **title_kw)
    ax1.set_xlabel("Count", **label_kw)
    ax1.tick_params(**tick_kw)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", color="#AAAAAA", fontsize=7)

    # ── 2. Model comparison bar ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    names  = list(results.keys())
    means  = [results[n]["mean_f1"] for n in names]
    stds   = [results[n]["std_f1"]  for n in names]
    colors = [COLORS[n] for n in names]
    bars2  = ax2.bar(range(len(names)), means, yerr=stds, color=colors,
                     capsize=5, edgecolor="#222", width=0.5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8, color="#AAAAAA")
    ax2.set_ylim(0, 1)
    ax2.set_facecolor("#1A1D27")
    ax2.set_title("5-Fold CV F1 Score (weighted)", **title_kw)
    ax2.set_ylabel("F1 Score", **label_kw)
    ax2.tick_params(**tick_kw)
    for bar, m in zip(bars2, means):
        ax2.text(bar.get_x() + bar.get_width()/2, m + 0.02,
                 f"{m:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")

    # ── 3. Training time ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    times = [results[n]["time"] for n in names]
    bars3 = ax3.bar(range(len(names)), times, color=colors, edgecolor="#222", width=0.5)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8, color="#AAAAAA")
    ax3.set_facecolor("#1A1D27")
    ax3.set_title("Training Time (5 folds, seconds)", **title_kw)
    ax3.set_ylabel("Seconds", **label_kw)
    ax3.tick_params(**tick_kw)

    # ── 4. Confusion matrix (best model) ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    y_pred = best_pipe.predict(X)
    cm = confusion_matrix(y, y_pred, labels=range(len(label_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax4.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax4.set_xticks(range(len(label_classes)))
    ax4.set_yticks(range(len(label_classes)))
    ax4.set_xticklabels(label_classes, rotation=45, ha="right", fontsize=7, color="#AAAAAA")
    ax4.set_yticklabels(label_classes, fontsize=7, color="#AAAAAA")
    ax4.set_title(f"Confusion Matrix — {best_name} (normalised)", **title_kw)
    ax4.set_facecolor("#1A1D27")
    for i in range(len(label_classes)):
        for j in range(len(label_classes)):
            val = cm_norm[i, j]
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if val > 0.5 else "#333")
    plt.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)

    # ── 5. Per-class F1 (best model) ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    report = classification_report(y, y_pred, target_names=label_classes, output_dict=True)
    per_class_f1 = [report[g]["f1-score"] for g in label_classes]
    colors_bar = plt.cm.RdYlGn([v for v in per_class_f1])
    bars5 = ax5.barh(label_classes, per_class_f1, color=colors_bar, edgecolor="#222")
    ax5.set_xlim(0, 1.05)
    ax5.set_facecolor("#1A1D27")
    ax5.set_title(f"Per-Class F1 — {best_name}", **title_kw)
    ax5.set_xlabel("F1 Score", **label_kw)
    ax5.tick_params(**tick_kw)
    for bar, val in zip(bars5, per_class_f1):
        ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", color="#AAAAAA", fontsize=7)

    # ── 6. Top TF-IDF features per genre ──────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    tfidf_vec = best_pipe.named_steps["tfidf"]
    clf       = best_pipe.named_steps["clf"]
    feature_names = np.array(tfidf_vec.get_feature_names_out())

    # For calibrated SVM grab base estimator
    try:
        coef = clf.coef_
    except AttributeError:
        try:
            coef = clf.calibrated_classifiers_[0].estimator.coef_
        except Exception:
            coef = None

    if coef is not None and len(label_classes) == coef.shape[0]:
        top_n     = 5
        n_genres  = len(label_classes)
        x_pos     = 0
        x_ticks, x_labels = [], []
        cmap = plt.cm.tab20(np.linspace(0, 1, n_genres))

        for gi, genre in enumerate(label_classes):
            top_idx   = np.argsort(coef[gi])[-top_n:][::-1]
            top_words = feature_names[top_idx]
            top_vals  = coef[gi][top_idx]
            xs = [x_pos + j for j in range(top_n)]
            ax6.bar(xs, top_vals / top_vals.max(), color=cmap[gi], edgecolor="#111", width=0.8)
            for xi, w in zip(xs, top_words):
                ax6.text(xi, -0.05, w, rotation=55, ha="right", va="top",
                         fontsize=6.5, color="#CCCCCC")
            x_ticks.append(x_pos + top_n / 2)
            x_labels.append(genre)
            x_pos += top_n + 1.5

        ax6.set_xticks(x_ticks)
        ax6.set_xticklabels(x_labels, fontsize=8, color="#DDDDDD", rotation=20, ha="right")
        ax6.set_facecolor("#1A1D27")
        ax6.set_title("Top-5 TF-IDF Features per Genre (normalised coefficient)", **title_kw)
        ax6.set_ylabel("Norm. coeff.", **label_kw)
        ax6.tick_params(axis="y", **tick_kw)
        ax6.axhline(0, color="#555", linewidth=0.5)
    else:
        ax6.text(0.5, 0.5, "Feature importance not available for this model",
                 ha="center", va="center", color="white", transform=ax6.transAxes)
        ax6.set_facecolor("#1A1D27")

    # Title
    fig.text(0.5, 0.965, "🎬  Movie Genre Classification — ML Pipeline Results",
             ha="center", color="white", fontsize=16, fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊  Plot saved → {out_path}")


# ─────────────────────────────────────────────
# 6. PREDICTION DEMO
# ─────────────────────────────────────────────
DEMO_PLOTS = [
    ("A group of astronauts travel to a distant planet and encounter a hostile alien race in a desperate battle for survival.",
     "sci-fi / action"),
    ("Two strangers fall in love during a summer in Paris while navigating complicated relationships and their own past.",
     "romance / drama"),
    ("A seasoned detective investigates a series of bizarre murders in a small coastal town, uncovering a decades-old conspiracy.",
     "mystery / crime"),
    ("Animated woodland creatures band together to save their enchanted forest from an evil sorcerer.",
     "animation / fantasy"),
    ("A mockumentary following three hapless roommates as they accidentally become amateur criminals in New York City.",
     "comedy / crime"),
]

def run_demo(best_pipe, le):
    print("\n" + "="*60)
    print("  PREDICTION DEMO")
    print("="*60)
    for plot, hint in DEMO_PLOTS:
        cleaned  = clean_text(plot)
        pred_idx = best_pipe.predict([cleaned])[0]
        proba    = best_pipe.predict_proba([cleaned])[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        pred_genre = le.classes_[pred_idx]
        print(f"\n  Plot:      {plot[:80]}…")
        print(f"  Hint:      {hint}")
        print(f"  Predicted: {pred_genre.upper()}  (confidence: {proba[pred_idx]:.1%})")
        top3 = [(le.classes_[i], proba[i]) for i in top3_idx]
        print(f"  Top-3:     " + "  |  ".join(f"{g} {p:.1%}" for g, p in top3))


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🎬  Movie Genre Classifier")
    print("    TF-IDF + Naive Bayes / Logistic Regression / Linear SVM\n")

    # Load
    df = load_data()
    print(f"  Dataset: {len(df):,} samples  |  {df['genre'].nunique()} genres")
    print(f"  Genres:  {sorted(df['genre'].unique())}")

    # Filter to known genres (in case IMDB has rare labels)
    known = df["genre"].value_counts()
    df = df[df["genre"].isin(known[known >= 10].index)].copy()

    # Preprocess
    print("\n  Cleaning text …")
    df["clean_plot"] = df["plot"].apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df["genre"])
    label_classes = le.classes_.tolist()
    X  = df["clean_plot"]

    # Train & evaluate
    results, best_name, best_pipe = evaluate_models(X, y, label_classes)

    # Full classification report
    y_pred = best_pipe.predict(X)
    print(f"\n{'='*60}")
    print(f"  FULL CLASSIFICATION REPORT — {best_name}")
    print(f"{'='*60}")
    print(classification_report(y, y_pred, target_names=label_classes))

    # Visualise
    print("  Generating plots …")
    plot_results(df, results, best_name, best_pipe, X, y, label_classes)

    # Demo
    run_demo(best_pipe, le)

    print("\n✅  Done!\n")


if __name__ == "__main__":
    main()
