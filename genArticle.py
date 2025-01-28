import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer from Hugging Face's transformers library
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc. for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

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

# Example usage
title = "How Artificial Intelligence is Changing the Future of Work"
article = generate_article(title)

# Print the generated article
print("Generated Article:\n")
print(article)
