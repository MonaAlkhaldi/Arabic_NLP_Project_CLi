from pathlib import Path
import click
import numpy as np

from utils.data_handler import load_csv, validate_columns, basic_dataset_summary
from utils.visualization import (
    ensure_vis_dir,
    summarize_lengths,
    plot_class_distribution_pie,
    plot_text_length_histogram,
    plot_top_words_bar,
)
from utils.arabic_text import load_stopwords, preprocess_text
from utils.embedding import build_tfidf, save_tfidf, build_model2vec, save_model2vec
from utils.training import get_models, train_and_eval, TrainResult
from utils.reporting import save_markdown_report
from joblib import dump


@click.group()
def cli():
    """Arabic NLP CLI Tool (Step-by-step pipeline)."""
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("text_col", type=str)
@click.argument("label_col", type=str)
@click.option(
    "--embed",
    type=click.Choice(["tfidf", "model2vec", "both"], case_sensitive=False),
    default="tfidf",
    show_default=True,
    help="Embedding method to use",
)
def pipeline(csv_path: str, text_col: str, label_col: str, embed: str):
    """
    Full NLP pipeline: Load → EDA → Preprocess → Embed → Train → Report
    """
    click.echo("Step 1: Loading and validating data...")
    try:
        project_root = Path(__file__).parent
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        # --- Step 1: Load & Validate ---
        df = load_csv(csv_path)
        validate_columns(df, text_col=text_col, label_col=label_col)
        summary = basic_dataset_summary(df, text_col=text_col, label_col=label_col)

        click.echo(f"Loaded: {summary['rows']} rows, {summary['cols']} columns")
        click.echo(f"Missing text rows: {summary['text_missing']}")
        click.echo(f"Number of classes: {summary['num_classes']}")
        click.echo("Step 1 finished successfully")

        # --- Step 2: EDA ---
        click.echo("Step 2: Running EDA...")
        vis_dir = ensure_vis_dir(project_root)

        # Class distribution pie
        pie_path = vis_dir / "eda_class_distribution_pie.png"
        plot_class_distribution_pie(df, label_col=label_col, out_path=pie_path)
        click.echo(f"Saved pie chart: {pie_path}")

        text_series = df[text_col].astype("string").fillna("")

        # Text length histogram (words)
        word_lengths = text_series.apply(lambda x: len(str(x).split()))
        word_stats = summarize_lengths(word_lengths)
        words_hist_path = vis_dir / "eda_text_length_words.png"
        plot_text_length_histogram(word_lengths, unit="words", out_path=words_hist_path)
        click.echo(f"Saved words histogram: {words_hist_path}")

        # Text length histogram (chars)
        char_lengths = text_series.apply(lambda x: len(str(x)))
        char_stats = summarize_lengths(char_lengths)
        chars_hist_path = vis_dir / "eda_text_length_chars.png"
        plot_text_length_histogram(char_lengths, unit="chars", out_path=chars_hist_path)
        click.echo(f"Saved chars histogram: {chars_hist_path}")

        # Top words bar chart
        top_words_path = vis_dir / "eda_top_words.png"
        plot_top_words_bar(text_series, out_path=top_words_path, top_k=20)
        click.echo(f"Saved top words chart: {top_words_path}")
        click.echo("Step 2 finished successfully")

        # --- Step 3: Preprocessing ---
        click.echo("Step 3: Preprocessing starting")
        stopwords_path = project_root / "resources" / "arabic_stopwords.txt"
        stopwords = load_stopwords(stopwords_path)

        df[text_col] = df[text_col].astype("string").fillna("").apply(
            lambda x: preprocess_text(x, stopwords)
        )

        preprocessed_path = outputs_dir / "preprocessed.csv"
        df.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")
        click.echo(f"✅ Preprocessed data saved to: {preprocessed_path}")

        texts = df[text_col].astype("string").fillna("").tolist()

        # --- Step 4: Embedding ---
        click.echo("Step 4: Creating embeddings...")
        X_tfidf = None
        emb_m2v = None

        if embed.lower() in ("tfidf", "both"):
            X_tfidf, vec = build_tfidf(texts, max_features=5000)
            X_path, vec_path = save_tfidf(X_tfidf, vec, outputs_dir)
            click.echo(f"🧠 TF-IDF shape: {X_tfidf.shape}")
            click.echo(f"💾 Saved TF-IDF: {X_path} | {vec_path}")

        if embed.lower() in ("model2vec", "both"):
            emb_m2v, m2v = build_model2vec(texts)
            emb_path, m2v_path = save_model2vec(emb_m2v, m2v, outputs_dir)
            click.echo(f"🧠 Model2Vec shape: {emb_m2v.shape}")
            click.echo(f"💾 Saved Model2Vec: {emb_path} | {m2v_path}")

        click.echo("✅ Step 4 done: Embeddings created successfully")

        # --- Step 5: Training + Reporting ---
        click.echo("Step 5: Training and reporting")
        reports_dir = outputs_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        models_dir = outputs_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        best_models = {}

        # TF-IDF training
        if embed.lower() in ("tfidf", "both"):
            y = df[label_col].astype("string").tolist()
            model_dict = get_models(["lr", "svm", "rf"])
            results_tfidf = train_and_eval(X_tfidf, y, model_dict)

            if not results_tfidf:
                raise click.ClickException("No TF-IDF models were successfully trained.")

            best_tfidf = results_tfidf[0]
            best_models["tfidf"] = best_tfidf

            report_path = save_markdown_report(
                out_dir=reports_dir,
                title="Training Report (TF-IDF)",
                dataset_info=summary,
                embedding_info=f"TF-IDF (max_features=5000, ngram_range=(1,2))",
                results=results_tfidf,
            )
            click.echo(f"📝 Saved TF-IDF report: {report_path}")
            click.echo(f"⭐ Best TF-IDF model: {best_tfidf.name.upper()} (acc={best_tfidf.accuracy:.4f})")

            # Save best TF-IDF model
            model_save_path = models_dir / f"best_model_tfidf.pkl"
            dump(best_tfidf.model, model_save_path)
            click.echo(f"💾 Saved best TF-IDF model: {model_save_path}")

        # Model2Vec training
        if embed.lower() in ("model2vec", "both"):
            y = df[label_col].astype("string").tolist()
            model_dict = get_models(["lr", "svm", "rf"])
            results_m2v = train_and_eval(emb_m2v, y, model_dict)

            if not results_m2v:
                raise click.ClickException("No Model2Vec models were successfully trained.")

            best_m2v = results_m2v[0]
            best_models["model2vec"] = best_m2v

            report_path2 = save_markdown_report(
                out_dir=reports_dir,
                title="Training Report (Model2Vec ARBERTv2)",
                dataset_info=summary,
                embedding_info="Model2Vec (JadwalAlmaa/model2vec-ARBERTv2)",
                results=results_m2v,
            )
            click.echo(f"📝 Saved Model2Vec report: {report_path2}")
            click.echo(f"⭐ Best Model2Vec model: {best_m2v.name.upper()} (acc={best_m2v.accuracy:.4f})")

            # Save best Model2Vec model
            model_save_path2 = models_dir / f"best_model_model2vec.pkl"
            dump(best_m2v.model, model_save_path2)
            click.echo(f"💾 Saved best Model2Vec model: {model_save_path2}")

        click.echo("✅ Step 5 done: Training and reports completed successfully")

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
