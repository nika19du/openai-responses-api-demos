import base64
import os
import sys

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8") # convert it to base64

base_image = encode_image("./image.jpg")

input_messages = [
    {
        "role": "user",
        "content":[
            {
                "type": "input_text",
                "text": "Please describe these images"
            },
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base_image}"
                    #"https://imgs.search.brave.com/b6OGOxD3O5mCFKRqAJy4XueEAKMrmt59dTTvpEPrUpg/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJzLmNvbS9p/bWFnZXMvaGlnaC9u/aXJ2YW5hLWt1cnQt/YW5kLWtyaXN0LTN0/dDV1cGo0aTUzMGtv/cWkuanBn" # passing url, not 64 encoded
            },
{
                "type": "input_image",
                "image_url": "https://imgs.search.brave.com/b6OGOxD3O5mCFKRqAJy4XueEAKMrmt59dTTvpEPrUpg/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJzLmNvbS9p/bWFnZXMvaGlnaC9u/aXJ2YW5hLWt1cnQt/YW5kLWtyaXN0LTN0/dDV1cGo0aTUzMGtv/cWkuanBn" # passing url, not 64 encoded
            },
        ]
    }
]

response = client.responses.create(
    model = "gpt-4o-mini",
    input = input_messages
)

print(response.output_text)