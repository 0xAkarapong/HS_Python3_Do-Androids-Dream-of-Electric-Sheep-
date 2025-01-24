import os
from mistralai import Mistral
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def get_llm_response_v1(user_message: str, function_list_str) -> str:
    prompt = f"""You are a helpful assistant that extracts function and interval for graph plotting.
    You can plot only the following functions: {function_list_str}.
    Allowed interval format is "from x_min to x_max".

    If the user asks to plot a graph, extract:
    - function_name (from the allowed list)
    - x_min (numeric value from the interval)
    - x_max (numeric value from the interval)

    Return the result in the format: "function_name,x_min,x_max"
    If the user wants to end the session, return "exit".
    If you cannot understand the request or the function is not in the allowed list, return "unknown".

    Example 1:
    User: Plot sine function from -3 to 3
    Assistant: sin(x),-3,3

    Example 2:
    User: I need a graph of y=x^2 between 0 and 10
    Assistant: x^2,0,10

    Example 3:
    User: bye
    Assistant: exit

    Example 4:
    User: Plot something complicated
    Assistant: unknown

    User message: {user_message}
    Assistant:
    """
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You are a graph plotting assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with Mistral LLM: {e}")
        return "error"


if __name__ == "__main__":
    allowed_functions_v1 = ["y=x", "y=x^2", "y=sin(x)", "y=cos(x)"]
    allowed_functions_str_v1 = ", ".join(allowed_functions_v1)

    while True:
        user_input = input("What graph do you want to plot? (or say 'bye' to exit): ")
        llm_response = get_llm_response_v1(user_input, allowed_functions_str_v1)
        print(llm_response)