from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from rouge_score import rouge_scorer
from sumy.summarizers.lex_rank import LexRankSummarizer  
from sumy.summarizers.luhn import LuhnSummarizer             
from sumy.summarizers.lsa import LsaSummarizer                 

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
    text = """
    Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.
    AI is a broad field of study that includes machine learning, natural language processing, robotics, and more.
    It is widely used in various industries such as healthcare, finance, and autonomous systems.
    AI-powered systems can process vast amounts of data and make predictions, enhancing efficiency and decision-making.
    """
    
    reference_summary = """
    AI simulates human intelligence in machines and is applied in fields like healthcare and finance to improve efficiency.
    """

    text_summarizer = TextSummarizer()
    methods = ["luhn", "lsa", "lex_rank", "text_rank"]
    for method in methods:
        generated_summary = text_summarizer.summarize_text(text, method, n_sentences=3)
        print("Generated Summary:", generated_summary)
    
        scores = text_summarizer.evaluate_summary(reference_summary, generated_summary)

        print("ROUGE Scores:")
        for metric, score in scores.items():
            print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-score={score.fmeasure:.4f}")
