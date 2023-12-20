from flask import Flask, render_template
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import spotipy.util as util
from openai import AzureOpenAI
import openai
import requests
from PIL import Image
from io import BytesIO

# Initialize app
app = Flask(__name__)

credentials = "spotify_keys.json"
with open(credentials, "r") as keys_file:
    api_tokens = json.load(keys_file)

client_id = api_tokens["client_id"]
client_secret = api_tokens["client_secret"]
redirectURI = api_tokens["redirect"]
username = api_tokens["username"]
authorize_endpoint = "https://accounts.spotify.com/authorize"

scope = 'user-read-private user-read-playback-state user-modify-playback-state playlist-modify-public user-library-read user-top-read'
token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirectURI)

# Create Spotify Object
sp = spotipy.Spotify(auth=token)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a route to handle the root URL
@app.route("/")
def home():
    return render_template("index.html")

# Define a route to handle the top tracks request
@app.route('/top-tracks')
def get_top_tracks():
    # Fetch user's top tracks from the Spotify API
    top_tracks = sp.current_user_top_tracks(time_range='short_term', limit=5)

    # Extract track names from the top tracks
    top_track_names = [track['name'] for track in top_tracks['items']]

    # Analyze the mood of the songs using ChatGPT
    mood_writeup = analyze_mood(top_track_names)

    # Print the mood writeup in the console (for debugging purposes)
    print("Mood Writeup:", mood_writeup)

    # Generate DALL·E-3 prompt based on the mood writeup
    dalle_3_prompt = generate_dalle_3_prompt(mood_writeup)
    print("DALL·E-3 Prompt:", dalle_3_prompt)

    # Generate DALL·E-3 image based on the prompt
    generated_image_url = generate_dalle_3_image(dalle_3_prompt)

    # Render the HTML template with the top tracks, mood writeup, DALL·E-3 prompt, and image
    return render_template('top_tracks.html', top_tracks=top_track_names, mood_writeup=mood_writeup, dalle_3_prompt=dalle_3_prompt, generated_image_url=generated_image_url)

def analyze_mood(song_list):
    # Generate a prompt asking ChatGPT to summarize the moods of the five songs into an overall feel of the music
    prompt = f"Summarize the moods of the following five songs into an overall feel of the music:\n"
    for song in song_list:
        prompt += f"- '{song}'\n"

    # Ask ChatGPT to generate the mood write-up
    mood_writeup = ask_chatgpt(prompt)

    return mood_writeup

def generate_dalle_3_prompt(mood_writeup):
    # Generate a DALL·E-3 prompt based on the reworded mood writeup
    prompt = f"Reword the following mood summary into a concise DALL·E-3 prompt:\n{mood_writeup}\nLimit the prompt to two sentences, use only one to four word strings separated by commas, avoid using song titles. Avoid using words such as Song descriptions, form the prompt from only one to four word strings"

    # Ask ChatGPT to reword the mood writeup into a concise DALL·E-3 prompt
    dalle_3_prompt = ask_chatgpt(prompt)

    return dalle_3_prompt

def generate_dalle_3_image(prompt):
    # Generate a DALL·E-3 image based on the prompt
    result = client.images.generate(
        model="dalle3",  # the name of your DALL-E 3 deployment
        prompt=prompt,
        n=1
    )

    # Get the image URL from the response
    image_url = json.loads(result.model_dump_json())["data"][0]["url"]

    return image_url

def ask_chatgpt(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that is well-versed in describing songs and their moods."},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    # Access the last message in the list of choices
    last_message = response.choices[-1].message

    # Check if the message has 'content' attribute before attempting to access it
    content = getattr(last_message, 'content', None)
    return content.strip() if content else "No response content available"

if __name__ == '__main__':
    app.run(debug=True)
