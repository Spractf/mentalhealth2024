import wave
import numpy as np
import speech_recognition as sr
import pyttsx3
import librosa
import pickle

# Load the trained model and vectorizer
with open("models\emotion_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)
# with open("models\tfidf_vectorizer.pkl", "rb") as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the audio storage
conversation_audio = "conversation_audio.wav"
audio_segments = []  # List to store audio segment data

# Function to combine audio segments
def save_combined_audio(output_path, audio_segments):
    if not audio_segments:
        return
    with wave.open(output_path, "wb") as output_wave:
        # Use parameters from the first segment
        output_wave.setparams(audio_segments[0]["params"])
        for segment in audio_segments:
            output_wave.writeframes(segment["frames"])

# Function for text-to-speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function for speech recognition and audio saving
def recognize_and_save_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("I am listening. Please share your thoughts.")
        try:
            audio = recognizer.listen(source, timeout=10)

            # Extract audio data and parameters directly from recognizer
            audio_data = audio.get_wav_data()
            audio_rate = source.SAMPLE_RATE
            audio_width = 2  # Assuming 16-bit audio
            audio_channels = 1  # Assuming mono audio

            # Store audio segment in memory
            audio_segments.append(
                {
                    "params": (audio_channels, audio_width, audio_rate, 0, "NONE", "not compressed"),
                    "frames": audio_data,
                }
            )

            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Can you repeat?")
            return None
        except sr.RequestError:
            speak("Sorry, my speech service is down.")
            return None

# Predict emotion from text
def predict_emotion_from_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    emotion_label = classifier.predict(text_tfidf)[0]

    # Map labels to emotion categories
    emotion_mapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    emotion = emotion_mapping.get(emotion_label, "neutral")
    return emotion

# Analyze audio features from combined audio
def analyze_combined_audio_features(audio_file):
    y, sr = librosa.load(audio_file)


    features = {
        "avg_pitch": np.mean(librosa.yin(y, sr=sr, fmin=30, fmax=500)),  # Adjust fmin/fmax as needed
        "max_pitch": np.max(librosa.yin(y, sr=sr, fmin=30, fmax=500)),
        "avg_rms": np.mean(librosa.feature.rms(y=y).T, axis=0)[0],
        "max_rms": np.max(librosa.feature.rms(y=y).T, axis=0)[0],
        "avg_zcr": np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0],
        "max_zcr": np.max(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]
    }


    features = {key: float(value) if isinstance(value, np.float32) else int(value) if isinstance(value, np.int64) else value for key, value in features.items()}

    return features

# Final health assessment
def assess_health(features):
    print("Analyzed features:", features)
    # Placeholder for speech output (integrate with a text-to-speech system, if needed)
    speak("Based on the overall conversation, here is what I noticed:")

    # Determine the emotion based on features
    if features["avg_pitch"] < 150 and features["avg_rms"] < 0.05:
        speak("You might be feeling low or sad. Remember, I'm here for you.")
        return "sad"
    elif features["max_pitch"] > 300:
        speak("There might be signs of stress or anger. Let's work through it together.")
        return "stress or anger"
    elif features["avg_rms"] > 0.1 and features["avg_zcr"] > 0.1:
        speak("You seem energetic or excited. That's great to hear!")
        return "excitement"
    elif features["avg_pitch"] > 200 and features["avg_rms"] < 0.08:
        speak("You seem to be feeling happy and positive!")
        return "happy"
    elif features["avg_pitch"] > 180 and features["avg_rms"] < 0.1:
        speak("There might be signs of hopefulness. Keep up the good energy!")
        return "hopeful"
    elif features["avg_pitch"] > 250 and features["avg_rms"] < 0.07:
        speak("You sound like you're feeling anxious. Take a deep breath and relax.")
        return "anxious"
    elif features["avg_rms"] > 0.15 and features["avg_zcr"] < 0.05:
        speak("You seem to be feeling tired or fatigued. It's important to rest.")
        return "tired"
    elif features["avg_pitch"] < 100 and features["avg_rms"] < 0.04:
        speak("It sounds like you might be feeling fearful. Take things slow and let me know how I can assist.")
        return "fearful"
    else:
        speak("Your state seems neutral. Let me know if you'd like to share more.")
        return "neutral"


# Main chatbot function
def mental_health_chatbot():
    speak("Hello! I'm here to check in with you. How are you feeling today?")
    user_text_responses = []

    while True:
        user_input = recognize_and_save_speech()
        if user_input:
            user_text_responses.append(user_input)
            if "exit" in user_input.lower() or "bye" in user_input.lower():
                speak("Thank you for sharing with me. Let me analyze the overall conversation.")
                break

    # Save combined audio to a file
    save_combined_audio(conversation_audio, audio_segments)

    # Analyze combined audio features
    features = analyze_combined_audio_features(conversation_audio)
    assess_health(features)

    # Perform text-based emotion prediction (optional)
    conversation_text = " ".join(user_text_responses)
    final_emotion = predict_emotion_from_text(conversation_text)
    speak(f"Based on our conversation, I sense a general emotion of {final_emotion}.")

# Run the chatbot
if __name__ == "__main__":
    mental_health_chatbot()
