from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_text(text):
    return [text[i:i+300] for i in range(0, len(text), 300)]

def get_answer(query, chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    index = similarity.argmax()
    return chunks[index]