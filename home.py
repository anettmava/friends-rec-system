import streamlit as st
import pymongo
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bson.binary import Binary
import pickle

# Debugging: Check if secrets are set
if "URI_URL" in st.secrets:
    st.write("MongoDB URI is set.")
else:
    st.error("MongoDB URI is not set in st.secrets.")

if "OPENAI_API_KEY" in st.secrets:
    st.write("OpenAI API key is set.")
else:
    st.error("OpenAI API key is not set in st.secrets.")

# MongoDB connection
try:
    uri = st.secrets["URI_URL"]
    client = pymongo.MongoClient(uri)
    db = client["friend_rec"]
    collection = db["responses"]  # like a table on sql
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")

# OpenAI API key for embeddings (replace with your own API key)
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"Error setting OpenAI API key: {e}")

st.title("Friends recommendation system")

st.divider()

st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimage.freepik.com%2Ffree-vector%2Fhand-drawn-valentine-s-day-penguins-couple_23-2148390371.jpg&f=1&nofb=1&ipt=1bc3afe7a7710f79006ea0810b3f5f62dda75dda3e29db6f12ddd4f3820dc79f&ipo=images")

st.subheader("Fill the questions below to get your techie match")

name = st.text_input("What is your name?")
hobby = st.text_input("What’s your favorite hobby or pastime?")
sports = st.text_input("Do you enjoy sports or exercise?")
traveling = st.text_input("Do you enjoy traveling?")
alone = st.text_input("Do you prefer spending time alone or with others?")
friend_group = st.text_input("Do you prefer small friend groups or big friend groups?")
fmovies = st.text_input("Do you watch a lot of movies or TV shows?")
description = st.text_input("Give us a short description about yourself")

def get_embedding(text):
    try:
        # Get the embedding for the response from OpenAI API 
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]  # The input should be a list of strings
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error getting embedding from OpenAI: {e}")
        return None

# Save responses on the mongo db data base
def save_response_to_db(responses, embedding):
    try:
        result = collection.insert_one({
            "responses": responses,
            "embedding": Binary(pickle.dumps(embedding))  # Store embeddings as binary
        })
        return result.inserted_id 
    except Exception as e:
        st.error(f"Error saving response to MongoDB: {e}")
        return None

def find_match(current_user_id, current_embedding):
    try:
        all_responses = list(collection.find())
        similarities = []

        for doc in all_responses:
            if doc['_id'] == current_user_id:
                continue
            stored_embedding = pickle.loads(doc['embedding'])
            similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]
            similarities.append((similarity, doc['responses']))

        # Sort by similarity (highest first)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        return similarities[0][1] if similarities else None
    except Exception as e:
        st.error(f"Error finding match: {e}")
        return None

if st.button("Submit"):
    if name and hobby and sports and traveling and alone and friend_group and fmovies and description:
        responses = {
            "name": name,
            "hobby": hobby,
            "sports": sports,
            "traveling": traveling,
            "alone": alone,
            "friend_group": friend_group,
            "fmovies": fmovies,
            "description": description
        }

        # Generate embeddings
        responses_text = " ".join(responses.values())
        actual_embedding = get_embedding(responses_text)

        if actual_embedding:
            # Add to mongodb
            current_user_id = save_response_to_db(responses, actual_embedding)

            if current_user_id:
                match = find_match(current_user_id, actual_embedding)

                if match:
                    st.success(f"Match found! Your match is {match['name']}")
                else:
                    st.warning("No match found yet, wait for someone to fill the form")
            else:
                st.error("Failed to save response to database.")
        else:
            st.error("Failed to generate embedding.")
    else:
        st.warning("Please fill all of the question blanks")