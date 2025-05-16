import json

# Sample dataset
data = {
  "questions": [
    {
      "category": "HR",
      "question": "Tell me about yourself",
      "answer": "I am a passionate learner, focusing on data science and machine learning. I believe in solving real-world problems with technology."
    },
    {
      "category": "HR",
      "question": "Why should we hire you?",
      "answer": "I have strong analytical skills, quick learning ability, and a passion for working in a team-oriented environment."
    },
    {
      "category": "Technical",
      "question": "What is overfitting in ML?",
      "answer": "Overfitting occurs when a machine learning model learns the training data too well, capturing noise and outliers, resulting in poor performance on new data."
    },
    {
      "category": "Technical",
      "question": "Explain the BERT model",
      "answer": "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model for NLP tasks, trained to predict missing words in sentences using context from both directions."
    }
  ]
}

with open("interview_questions.json","w") as f:
    json.dump(data, f, indent=4)

from sentence_transformers import SentenceTransformer
import faiss
import json

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
with open("interview_questions.json", "r") as f:
    data = json.load(f)


# Prepare questions for embedding
questions = [entry['question'] for entry in data['questions']]
answers = [entry['answer'] for entry in data['questions']]


#load FAISS index
index = faiss.read_index(f"faiss_index.idx")


#Inference function
def get_answer(user_question):
    question_embedding = model.encode([user_question])
    D, I = index.search(question_embedding, k=1) #Top 1 result
    match_index = I[0][0]
    similarty_score = D[0][0]
    print(f"similarty score: {similarty_score:.2f}")
    return answers[match_index]


#Example
user_q = input("Ask a question: ")
response = get_answer(user_q)
print("\nBot's Answer:", response)