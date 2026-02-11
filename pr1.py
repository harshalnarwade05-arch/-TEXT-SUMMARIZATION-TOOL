import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_text(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices.sort()

    summary = " ".join(sentences[i] for i in top_indices)
    return summary


input_text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.
It enables machines to understand, interpret, and generate text.
NLP is widely used in applications such as chatbots, sentiment analysis, search engines, and text summarization.
As digital content continues to grow, automatic text summarization has become essential for quickly understanding large documents.
"""

summary = summarize_text(input_text, num_sentences=2)

print("Original Text:\n", input_text)
print("\nSummary:\n", summary)
