import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def chat_loop():
    current_response_id = None

    while True:
        # get user input
        user_input = input('Enter your message: ')

        if user_input.lower() in ['exit','quit', 'bye']:
            print('Goodbye!')
            break

        response = client.responses.create(
             model = "gpt-4o-mini",
             input = user_input,
            previous_response_id = current_response_id
        )

        current_response_id = response.id

        # print the response
        print("Bot: ", response.output_text)

if __name__ == '__main__':
    chat_loop()
