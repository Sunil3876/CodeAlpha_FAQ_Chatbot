import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK data download (ek baar run hoga)
nltk.download('punkt', quiet=True)

def load_faqs(filepath):
    questions = []
    answers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                q, a = line.strip().split('|', 1)
                questions.append(q)
                answers.append(a)
    return questions, answers

def get_best_response(user_query, questions, answers):
    # If there are no FAQs loaded, return a friendly fallback
    if not questions:
        return "Sorry, I don't have any FAQs available right now."

    all_texts = questions + [user_query]

    # TF-IDF ka use karke vectorize karna
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # If there's only the user query (no FAQ texts), bail out
    if tfidf_matrix.shape[0] < 2:
        return "Sorry, I don't have enough data to answer that right now."

    # Cosine similarity match karna
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Guard against empty similarity results
    if cosine_sim.size == 0:
        return "Sorry, I don't understand that question based on my FAQs."

    best_match_idx = int(cosine_sim.argmax())
    best_match_score = float(cosine_sim[0, best_match_idx])

    if best_match_score > 0.2:
        return answers[best_match_idx]
    else:
        return "Sorry, I don't understand that question based on my FAQs."

if __name__ == "__main__":
    print("Loading FAQs...")
    questions, answers = load_faqs('data.txt')
    print("Chatbot is ready! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Best matching answer ko chatbot response mein dikhana
        response = get_best_response(user_input, questions, answers)
        print(f"Chatbot: {response}\n")
