import glob

import pandas as pd

from rouge_score import rouge_scorer

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

from sumy.summarizers.lsa import LsaSummarizer                 
from sumy.summarizers.luhn import LuhnSummarizer             
from sumy.summarizers.lex_rank import LexRankSummarizer  
from sumy.summarizers.text_rank import TextRankSummarizer

class TextSummarizer:

    def __init__(self):
        self.summarizers = {
            "luhn": self.luhn_summarizer,
            "lsa": self.lsa_summarizer,
            "lex_rank": self.lex_rank_summarizer,
            "text_rank": self.text_rank_summarizer,
        }

    def summarize_text(self, text, method, n_sentences=3):
        summarizer = self.summarizers[method]
        summary = summarizer(text, n_sentences=n_sentences)
        return summary

    def luhn_summarizer(self, text, n_sentences):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer_lex = LuhnSummarizer()                      
        summary = summarizer_lex(parser.document, n_sentences)              
        return " ".join(str(sentence) for sentence in summary)

    def lsa_summarizer(self, text, n_sentences):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer_lex = LsaSummarizer()                      
        summary = summarizer_lex(parser.document, n_sentences)              
        return " ".join(str(sentence) for sentence in summary)


    def lex_rank_summarizer(self, text, n_sentences):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer_lex = LexRankSummarizer()                      
        summary = summarizer_lex(parser.document, n_sentences)              
        return " ".join(str(sentence) for sentence in summary)


    def text_rank_summarizer(sellf, text, n_sentences):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, n_sentences)
        return " ".join(str(sentence) for sentence in summary)

    def evaluate_summary(self, reference_summary, generated_summary):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        return scores

class BBCNews_Dataset:

    def __init__(self, filenames):
        self.filenames = glob.glob(filenames)

    def __len__(self):
        return len(self.filenames)
    
    def get(self, i):
        article_filename = self.filenames[i]
        with open(article_filename) as f:
            article = f.read()

        summary_filename = article_filename.replace("articles", "summaries")
        with open(summary_filename) as f:
            summary = f.read()

        return {
            "article": article,
            "summary": summary
        }
    
class CNNNews_Dataset:

    def __init__(self, filenames):
        self.df = pd.concat(filenames)

    def __len__(self):
        return len(self.df)
    
    def get(self, i):
        row = self.df.iloc[i]
        return {
            "article": row["article"],
            "summary": row["highlights"]
        }


if __name__ == "__main__":
    # filenames = [
    #     pd.read_csv("datasets/text_summary/cnn_news/train.csv"),
    #     pd.read_csv("datasets/text_summary/cnn_news/test.csv"),
    #     pd.read_csv("datasets/text_summary/cnn_news/validation.csv"),
    # ]
    # dataset = CNNNews_Dataset(filenames=filenames)

    filenames = "datasets/text_summary/bbc_news/articles/*/*.txt"
    dataset = BBCNews_Dataset(filenames=filenames)

    data = dataset.get(0)
    text = data["article"]
    reference_summary = data["summary"]
    print(text)

    text_summarizer = TextSummarizer()
    methods = ["luhn", "lsa", "lex_rank", "text_rank"]
    for method in methods:
        generated_summary = text_summarizer.summarize_text(text, method, n_sentences=3)
        print("Generated Summary:", generated_summary)
    
        scores = text_summarizer.evaluate_summary(reference_summary, generated_summary)

        print("ROUGE Scores:")
        for metric, score in scores.items():
            print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-score={score.fmeasure:.4f}")
