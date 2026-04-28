import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text.lower()

def get_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)
    return " ".join([str(sentence) for sentence in summary])