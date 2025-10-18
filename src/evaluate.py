import numpy as np
import sacrebleu
from transformers import pipeline, T5TokenizerFast, TFT5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

def token_f1(gold, pred):
    g_tokens = gold.split()
    p_tokens = pred.split()
    common = set(g_tokens) & set(p_tokens)
    if len(common) == 0:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0.0

def evaluate_model(model_dir="models/t5_medqa_final", test_path="data/processed/test.csv"):
    tokenizer = T5TokenizerFast.from_pretrained(model_dir)
    model = TFT5ForConditionalGeneration.from_pretrained(model_dir)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="tf")

    test_df = pd.read_csv(test_path).sample(200, random_state=42)
    refs, hyps = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred = gen("question: " + row["question"], max_length=128, num_return_sequences=1)[0]["generated_text"]
        refs.append([row["answer"]])
        hyps.append(pred)

    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs))[0])
    f1 = np.mean([token_f1(r[0], h) for r, h in zip(refs, hyps)])

    print(f"BLEU: {bleu.score:.2f}")
    print(f"Token F1: {f1:.4f}")

if __name__ == "__main__":
    evaluate_model()
