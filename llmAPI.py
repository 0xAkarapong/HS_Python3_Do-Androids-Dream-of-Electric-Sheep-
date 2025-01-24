import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-2409"

client = Mistral(api_key=api_key)

while True:
    input_message = input("Enter your query: ")
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": f"{input_message}",
            },
        ]
    )
    print(chat_response.choices[0].message.content)