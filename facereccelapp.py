from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import face_recognition
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

def find_celebrity_lookalike(user_image_path, all_celebrity_names, all_celebrity_encodings):
    """
    Compare the user's face encoding to all celebrity encodings and return the closest match.

    Parameters:
    - user_image_path: Path to the user's image.
    - all_celebrity_names: List of celebrity names.
    - all_celebrity_encodings: List of face encodings for the celebrities.

    Returns:
    - Closest celebrity match.
    """

    # Load the user's image and compute the face encoding
    user_image = face_recognition.load_image_file(user_image_path)
    user_encoding = face_recognition.face_encodings(user_image)

    # Check if a face is detected in the user's image
    if not user_encoding:
        return "No face detected in the image."

    # Find the face distances for the user's face encoding with all celebrity encodings
    face_distances = face_recognition.face_distance(all_celebrity_encodings, user_encoding[0])

    # Find the index of the celebrity with the smallest face distance
    best_match_index = np.argmin(face_distances)

    # Return the name of the celebrity with the closest match
    return all_celebrity_names[best_match_index]

# Load the previously saved celebrity encodings
with open('/content/celebrity_encodings_batch1.pkl', 'rb') as file:
    celeb_names_batch1, celeb_encodings_batch1 = pickle.load(file)
with open('/content/celebrity_encodings_next90.pkl', 'rb') as file:
    celeb_names_next90, celeb_encodings_next90 = pickle.load(file)

# Combine encodings and names from both batches
all_names = celeb_names_batch1 + celeb_names_next90
all_encodings = celeb_encodings_batch1 + celeb_encodings_next90

@app.route('/find_lookalike', methods=['POST'])
def find_lookalike():
    try:
        uploaded_image = request.files['user_image']
        if uploaded_image:
            user_image_path = '/path/to/uploaded/image'  # Update this with the actual path
            lookalike = find_celebrity_lookalike(user_image_path, all_names, all_encodings)
            return jsonify({'lookalike': lookalike})
        else:
            return jsonify({'error': 'No image uploaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
