# Friends Recommendation System 🫂

This is a  web application that helps users find friendship matches based on their responses to a set of personality-related questions. The system utilizes OpenAI embeddings and cosine similarity to determine the most compatible matches from a MongoDB database.

---

##  Features

- **Interactive UI**: Users fill out a form about their personality and preferences.
- **AI-powered Matching**: Uses OpenAI's text embeddings and cosine similarity to find the closest match.
- **MongoDB Integration**: Stores user responses and embeddings in a MongoDB database.
- **Real-time Feedback**: Displays matching results instantly or prompts users to wait for more entries.

---
## How It Works
- **User Input:** The user answers questions about their personality, hobbies, and preferences.
- **Embedding Generation:** The responses are converted into text embeddings using OpenAI’s text-embedding-ada-002 model.
- **Data Storage:** Responses and embeddings are stored in MongoDB.
- **Matching Algorithm:** Uses cosine similarity to compare embeddings and find the most similar user.
- **Results Display:** The system suggests the best friendship match or informs the user to wait for more entries.

---
## Technologies Used
- Python
- Streamlit (UI Framework)
- MongoDB (Database)
- Pymongo (MongoDB Client)
- OpenAI API (Embeddings)
- Scikit-learn (Cosine Similarity)
- Pickle & BSON (Embedding Serialization)
