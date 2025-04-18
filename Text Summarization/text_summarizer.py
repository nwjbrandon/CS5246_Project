import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import pipeline
from rouge_score import rouge_scorer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer                 
from sumy.summarizers.luhn import LuhnSummarizer             
from sumy.summarizers.lex_rank import LexRankSummarizer  
from sumy.summarizers.text_rank import TextRankSummarizer


nltk.download('punkt_tab')


class TextSummarizer:

    def __init__(self):
        self.summarizers = {
            "BART": self.facebook_bart_summarizer,
            "Luhn": self.luhn_summarizer,
            "LSA": self.lsa_summarizer,
            "LexRank": self.lex_rank_summarizer,
            "TextRank": self.text_rank_summarizer,
        }
        self.algorithms = [summarizer for summarizer in self.summarizers]

        self.summarizer_luhn = LuhnSummarizer()   
        self.summarizer_lsa = LsaSummarizer()   
        self.summarizer_lexrank = LexRankSummarizer()   
        self.summarizer_textrank = TextRankSummarizer()
        self.summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_text(self, text, method):
        summarizer = self.summarizers[method]
        summary = summarizer(text)
        return summary

    def luhn_summarizer(self, text, n_sentences=5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer_luhn(parser.document, n_sentences)              
        return " ".join(str(sentence) for sentence in summary)

    def lsa_summarizer(self, text, n_sentences=5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer_lsa(parser.document, n_sentences)              
        return " ".join(str(sentence) for sentence in summary)


    def lex_rank_summarizer(self, text, n_sentences=5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer_lexrank(parser.document, n_sentences)              
        return " ".join(str(sentence) for sentence in summary)


    def text_rank_summarizer(self, text, n_sentences=5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer_textrank(parser.document, n_sentences)
        return " ".join(str(sentence) for sentence in summary)
    
    def facebook_bart_summarizer(self, text, min_length=30, max_length=130, max_input_length=1024):
        input_length = len(text.split())
        max_length = input_length if input_length < max_length else max_length        
        min_length = min(min_length, max_length)
        return self.summarizer_bart(text[:max_input_length], min_length=min_length, max_length=max_length, do_sample=False)[0]["summary_text"]


def rouge_score_metric(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

def evaluate_text_summarizer(text_summarizer, dataset, text_key, summary_key):
    full_metrics = []

    for algorithm in text_summarizer.algorithms:
        rouge1_precisions = []
        rouge1_recalls = []
        rouge1_fmeasures = []

        rouge2_precisions = []
        rouge2_recalls = []
        rouge2_fmeasures = []

        rougeL_precisions = []
        rougeL_recalls = []
        rougeL_fmeasures = []

        for data in tqdm(dataset):
            text = data[text_key]
            reference_summary = data[summary_key]

            # Skip if the text is too short
            input_length = len(text.split())
            if input_length < 300:
                continue

            generated_summary = text_summarizer.summarize_text(text, algorithm)
            rouge_scores = rouge_score_metric(reference_summary, generated_summary)
            
            for metric, score in rouge_scores.items():
                if metric == "rouge1":
                    rouge1_precisions.append(score.precision)
                    rouge1_recalls.append(score.recall)
                    rouge1_fmeasures.append(score.fmeasure)
                if metric == "rouge2":
                    rouge2_precisions.append(score.precision)
                    rouge2_recalls.append(score.recall)
                    rouge2_fmeasures.append(score.fmeasure)
                if metric == "rougeL":
                    rougeL_precisions.append(score.precision)
                    rougeL_recalls.append(score.recall)
                    rougeL_fmeasures.append(score.fmeasure)

        full_metrics.append([algorithm, "Rouge1", np.mean(score.precision), np.mean(score.recall), np.mean(score.fmeasure)])
        full_metrics.append([algorithm, "Rouge2", np.mean(score.precision), np.mean(score.recall), np.mean(score.fmeasure)])
        full_metrics.append([algorithm, "RougeL", np.mean(score.precision), np.mean(score.recall), np.mean(score.fmeasure)])

    df = pd.DataFrame(full_metrics, columns=["Algorithm", "Metric", "P", "R", "F1"])
    return df
