from rest_framework.views import APIView
from rest_framework.response import Response
from sentence_transformers import CrossEncoder
import numpy as np  # Import numpy to handle numpy arrays

# Load the CrossEncoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Scaling function
def scale_to_percentage(score, min_score=-5, max_score=5, min_range=0, max_range=100):
    normalized_score = (score - min_score) / (max_score - min_score)
    scaled_score = normalized_score * (max_range - min_range) + min_range
    scaled_score = max(scaled_score, 0)
    return f"{round(scaled_score)}%"

# API View
class SemanticSearchView(APIView):
    def post(self, request):
        question = request.data.get('question')
        answers = request.data.get('answers', [])

        if not question or not answers:
            return Response({"error": "Question and answers are required."}, status=400)

        # Generate query pairs and predict scores
        query_pairs = [(question, answer) for answer in answers]
        scores = cross_encoder.predict(query_pairs)
        percentage_scores = [scale_to_percentage(score) for score in scores]

        # Find the best match
        best_match_index = np.argmax(scores)
        best_match_answer = answers[best_match_index]
        best_match_score = percentage_scores[best_match_index]
        
        # Return the results
        return Response({
            "scaled_scores": percentage_scores,
            "best_match":{
                "answer": best_match_answer,
                "score": best_match_score
            }
        })
