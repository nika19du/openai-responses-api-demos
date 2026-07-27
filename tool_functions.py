import json
import os

import requests
from openai import OpenAI
from dotenv import  load_dotenv
load_dotenv()

from pydantic import BaseModel, ConfigDict


# Tool Functions
def get_todo_list():
    return [
        {
            "id": 1,
            "title": "Buy bread"
        },
        {
            "id": 2,
            "title": "Buy milk"
        }
    ]

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()

    return data['current']['temperature_2m']

# Pydantic schemas
class GetTodoListParams(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    pass

class GetWeatherParams(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    latitude: float
    longitude: float

# Generate schemas, create a schema
todo_list_schema = GetTodoListParams.model_json_schema()

weather_schema = GetWeatherParams.model_json_schema()

tools = [
    {
        "type": "function",
        "name": "get_todo_list",
        "description": "Get the list of todos", # this will tell the ai model when to use this tool
        "parameters": todo_list_schema,
        "strict": True
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a given latitude and longitude",
        "parameters": weather_schema,
        "strict": True
    }
]

# Maps tool names to actual Python functions
tool_functions = {
    "get_todo_list": get_todo_list,
    "get_weather": get_weather
}

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

input_messages = [
    {
        "role": "user",
        "content": "What is the current weather in Plovdiv? Also, what are my todos?"
    }
]

while True:
    tool_call_response = client.responses.create(
        model = "gpt-4o-mini",
        input = input_messages,
        tools = tools
    )
    # preserve everything returned by the model
    input_messages.extend(tool_call_response.output)

    function_calls = [item for item in tool_call_response.output if item.type == "function_call"]

    # No tool calls means the final text is ready
    if not function_calls:
        print(tool_call_response.output_text)
        break

    # execute every function call returned in this response
    for function_call in function_calls:
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)

        print(f"Calling {function_name} with args: {arguments}")

        function = tool_functions.get(function_name)
        if function is None:
            tool_result = {
                "error": f"Unknown function call: {function_name}"
            }
        else:
            try:
                tool_result = function(**arguments)
            except Exception as error:
                tool_result = {
                    "error": str(error)
                }

        input_messages.append({
            "type": "function_call_output",
            "call_id": function_call.call_id,
            "output": json.dumps(tool_result)
        })