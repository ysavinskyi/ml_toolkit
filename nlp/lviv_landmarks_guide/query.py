import numpy as np

from sentence_transformers import SentenceTransformer, util
from nlp.lviv_landmarks_guide.context import lviv_landmarks_info, lviv_landmarks_intents


class Query:
    """
    Query class for handling user queries about Lviv landmarks.
    """
    print('<<<< Loading the model... >>>>')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('<<<< Model ready! >>>>')

    context_embeddings = {}
    for landmark, info in lviv_landmarks_info.items():
        context_embeddings[landmark] = model.encode(info["description"], convert_to_tensor=True)

    intent_embeddings = {}
    for intent, phrases in lviv_landmarks_intents.items():
        embeddings = model.encode(phrases)
        intent_embeddings[intent] = np.mean(embeddings, axis=0)  # Use the mean embedding for each intent

    def __init__(self, query):
        """
        Initialize the Query object with the user query.
        """
        self._query = query.lower()
        self._query_type = ''
        self._landmark_match = ''
        self._classify_query()

    def _classify_query(self):
        """
        Classifies the query intent using cosine similarity with precomputed intent embeddings.
        """
        # Compute the embedding for the user query
        query_embedding = self.model.encode(self._query)

        # Calculate cosine intent similarities with each intent embedding
        intent_similarities = {
            intent: util.pytorch_cos_sim(query_embedding, embedding).item()
            for intent, embedding in self.intent_embeddings.items()
        }
        best_intent_match = max(intent_similarities, key=intent_similarities.get)

        # Check if the query is about a specific landmark
        landmark_similarities = {
            landmark: util.pytorch_cos_sim(query_embedding, desc_emb).item()
            for landmark, desc_emb in self.context_embeddings.items()
        }
        best_landmark_match = max(landmark_similarities, key=landmark_similarities.get)

        if landmark_similarities[best_landmark_match] > intent_similarities[best_intent_match]:  # Threshold for specific landmark match
            self._query_type = "specific"
            for landmark in landmark_similarities:
                if landmark in self._query:
                    self._landmark_match = landmark
            else:
                self._landmark_match = best_landmark_match
        else:
            self._query_type = best_intent_match

    def _response_specific(self):
        """
        Answers the query using cosine similarity to find the best matching landmark.
        :returns: The answer about specific place
        """
        best_match = self._landmark_match
        response = f"Based on your interest in '{self._query}', you might want to check out '{best_match}': \n\n" \
                   f"{lviv_landmarks_info[best_match]['description']}\n" \
                   f"It is located at {lviv_landmarks_info[best_match]['location']}"

        return response

    def process(self):
        """
        Processes the query and returns an appropriate response based on the classified intent.
        :returns: Answer to query from a context
        """
        intent = self._query_type

        if intent == "general":
            response = "Top places to visit in Lviv include:\n"
            for landmark, info in lviv_landmarks_info.items():
                if "must-see" in info["tags"]:
                    response += f"- {landmark}\n"

        elif intent == "castles":
            response = "Castles in Lviv include:\n"
            for landmark, info in lviv_landmarks_info.items():
                if "castle" in info["tags"]:
                    response += f"- {landmark}\n"

        elif intent == "churches":
            response = "Cathedrals and churches in Lviv include:\n"
            for landmark, info in lviv_landmarks_info.items():
                if "church" in info["tags"] or "cathedral" in info["tags"]:
                    response += f"- {landmark}\n"

        elif intent == "art":
            response = "Art in Lviv is represented by:\n"
            for landmark, info in lviv_landmarks_info.items():
                if "art" in info["tags"]:
                    response += f"- {landmark}\n"

        elif intent == "entertainment":
            response = "Entertainment options in Lviv include:\n"
            for landmark, info in lviv_landmarks_info.items():
                if "entertainment" in info["tags"]:
                    response += f"- {landmark}\n"

        elif intent == "follow-up":
            response = "Other interesting places you might consider visiting are:\n"
            for landmark, info in lviv_landmarks_info.items():
                if "must-see" not in info["tags"]:
                    response += f"- {landmark}\n"

        elif intent == "specific":
            response = self._response_specific()

        else:
            response = "I'm sorry, I couldn't understand your query. Could you please specify your request?"

        return response
