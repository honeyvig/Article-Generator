import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from googleapiclient.discovery import build

# Define API key (replace with your own API key)
API_KEY = 'AIzaSyBhFzwf9cU7q-RcH23kEXDtcgZ4s56J7H4'  # Replace with your actual YouTube API Key

# Channel ID (tanmayteaches) from the provided YouTube URL
CHANNEL_ID = 'UCqufIGIYauviVaKyJUzKvQw'

# Build the YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Load pre-trained model and tokenizer from Hugging Face's transformers library
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc. for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Function to get videos from the channel
def get_video_titles(channel_id):
    video_titles = []
    
    next_page_token = None
    while True:
        # Make the API request to search for videos on the channel
        search_request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,  # Get up to 50 results per request
            pageToken=next_page_token,
            type="video"  # We only want video results
        )
        search_response = search_request.execute()

        # Extract video titles from the search results
        for item in search_response['items']:
            video_title = item['snippet']['title']
            video_titles.append(video_title)
        
        # Get the next page token, if any
        next_page_token = search_response.get('nextPageToken')

        # If there are no more pages of results, break out of the loop
        if not next_page_token:
            break

    return video_titles

# Function to generate an article based on a given title
def generate_article(title, max_length=300):
    # Encode the title as input to the model
    input_ids = tokenizer.encode(title, return_tensors='pt')

    # Generate text based on the input title
    with torch.no_grad():
        output = model.generate(input_ids, 
                                max_length=max_length, 
                                num_return_sequences=1, 
                                no_repeat_ngram_size=2, 
                                top_k=50, 
                                top_p=0.95, 
                                temperature=1.0, 
                                early_stopping=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the original title part from the generated text
    article = generated_text[len(title):]
    
    return article.strip()

# Create the folder to store the generated articles
folder_name = "generatearticles"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Fetch video titles from the channel
titles = get_video_titles(CHANNEL_ID)

# Generate and save an article for each video title
for idx, title in enumerate(titles, start=1):
    print(f"Generating article for video {idx} ({title})...")
    
    # Generate the article for the given title
    article = generate_article(title)
    
    # Create a file for each generated article and save it in the folder
    file_name = f"{folder_name}/article_{idx}_{title[:50]}.txt"  # Limit file name length
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(article)
    
    print(f"Article saved for video {idx} ({title})")

print("Article generation process completed.")
