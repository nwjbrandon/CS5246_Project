from datasets import load_dataset

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


if __name__ == "__main__":
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
    data = ds["train"][0]

    article = data["article"]
    summary_ref = data["highlights"]
    print(article)

    text_summarizer = TextSummarizer()
    methods = ["luhn", "lsa", "lex_rank", "text_rank"]
    for method in methods:
        summary_gen = text_summarizer.summarize_text(article, method, n_sentences=3)
        print("Generated Summary:", summary_gen)
    
        scores = text_summarizer.evaluate_summary(summary_ref, summary_gen)

        print("ROUGE Scores:")
        for metric, score in scores.items():
            print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-score={score.fmeasure:.4f}")
