from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from rouge_score import rouge_scorer
from sumy.summarizers.lex_rank import LexRankSummarizer  
from sumy.summarizers.luhn import LuhnSummarizer             
from sumy.summarizers.lsa import LsaSummarizer                 

def summarize_text_luhn(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_lex = LuhnSummarizer()                      
    summary= summarizer_lex(parser.document, num_sentences)              
    return " ".join(str(sentence) for sentence in summary)

def summarize_text_lsa(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_lex = LsaSummarizer()                      
    summary= summarizer_lex(parser.document, num_sentences)              
    return " ".join(str(sentence) for sentence in summary)


def summarize_text_lex(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_lex = LexRankSummarizer()                      
    summary= summarizer_lex(parser.document, num_sentences)              
    return " ".join(str(sentence) for sentence in summary)


def summarize_text(text, num_sentences=3):
    """Summarizes the input text using TextRank."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def evaluate_summary(reference_summary, generated_summary):
    """Evaluates the generated summary using ROUGE scores."""
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
    
    generated_summary = summarize_text_lsa(text, num_sentences=2)
    print("Generated Summary:", generated_summary)
    
    scores = evaluate_summary(reference_summary, generated_summary)
    print("\nROUGE Scores:")
    for metric, score in scores.items():
        print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-score={score.fmeasure:.4f}")
