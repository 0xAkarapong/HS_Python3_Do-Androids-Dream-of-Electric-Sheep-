import os
from mistralai import Mistral
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import re

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def get_llm_response_v1(user_message: str, function_list_str: str) -> str:
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

def parse_llm_response_v1(llm_response: str) -> tuple[str, str | None, float | None, float | None]:
    if llm_response == "exit":
        return "exit", None, None, None
    if llm_response == "unknown" or llm_response == "error":
        return "unknown", None, None, None

    try:
        parts = llm_response.split(',')
        function_name = parts[0].strip()
        x_min = float(parts[1])
        x_max = float(parts[2])
        return "plot", function_name, x_min, x_max
    except Exception:
        return "unknown", None, None, None

def plot_graph_v1(function_name: str, x_min: float, x_max: float) -> bool:
    x = np.linspace(x_min, x_max, 400)
    if function_name == 'x':
        y = x
    elif function_name == 'x^2':
        y = x ** 2
    elif function_name == 'sin(x)':
        y = np.sin(x)
    elif function_name == 'cos(x)':
        y = np.cos(x)
    else:
        print(f"Error: Unknown function '{function_name}' (internal error).")
        return False
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(f'Plot of y={function_name} from {x_min} to {x_max}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    return True

def get_llm_response_v2(user_message:str) -> str:
    prompt = f"""
    You are a helpful assistant that extracts function and interval for graph plotting.
    You can plot:
    - Polynomials of power 1 to 4 (e.g., "x^3 - 2x + 1").
    - Scaled sine and cosine functions (e.g., "sin(3x)", "cos(0.5x)").

    For polynomials, extract the coefficients as a list, starting from the highest power.
    For scaled trig functions, extract the function name (sin or cos) and the scale factor 'k'.

    Allowed interval format is "from x_min to x_max" or "between x_min and x_max".

    If the user asks to plot a graph, extract:
    - function_type (polynomial, sin, cos)
    - function_parameters (list of coefficients for polynomial, scale factor 'k' for trig)
    - x_min (numeric value from the interval)
    - x_max (numeric value from the interval)

    Return the result in the format: "function_type,function_parameters,x_min,x_max"

    For polynomials, function_parameters should be a comma-separated list of coefficients.
    For sin(kx) and cos(kx), function_parameters should be just the value of k.

    If the user wants to end the session, return "exit".
    If you cannot understand the request, return "unknown".

    Example 1:
    User: Plot x^2 - x + 2 from -2 to 3
    Assistant: polynomial,[1,-1,2],-2,3

    Example 2:
    User: Graph of sin(2x) between 0 and 2pi
    Assistant: sin,2,0,6.28

    Example 3:
    User: plot cos(0.5x) from -pi to pi
    Assistant: cos,0.5,-3.14,3.14

    Example 4:
    User: Plot x^4 + 1
    Assistant: polynomial,[1,0,0,0,1],-10,10 (Default interval if not specified)

    Example 5:
    User: bye bye
    Assistant: exit

    Example 6:
    User: draw something else
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

def parse_llm_response_v2(llm_response: str) -> tuple[str, str | None, str | None, float | None, float | None]:
    if llm_response == "exit":
        return "exit", None, None, None, None
    if llm_response == "unknown" or llm_response == "error":
        return "unknown", None, None, None, None

    try:
        parts = llm_response.split(',', 3)
        function_type = parts[0].strip()
        function_params_str = parts[1].strip()
        interval_part = parts[2]

        interval_match = re.search(r"([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)",
                                   interval_part)
        if interval_match:
            x_min = float(interval_match.group(1))
            x_max = float(interval_match.group(2))
        else:
            print("Warning: Could not parse interval correctly from LLM response.")
            return "unknown", None, None, None, None

        if function_type == 'polynomial':
            function_params = []
            try:
                coeff_str_list = function_params_str.strip('[]').split(',')
                function_params = [float(coeff.strip()) for coeff in coeff_str_list]
            except:
                print("Warning: Could not parse polynomial coefficients correctly.")
                return "unknown", None, None, None, None
        elif function_type in ['sin', 'cos']:
            try:
                function_params = float(function_params_str)  # k value
            except:
                print("Warning: Could not parse scale factor correctly.")
                return "unknown", None, None, None, None
        else:
            print("Warning: Unknown function type from LLM.")
            return "unknown", None, None, None, None

        return "plot", function_type, function_params, x_min, x_max

    except Exception as e:
        print(f"Parsing error: {e}")
        return "unknown", None, None, None, None


def plot_graph_v2(function_type: str, function_parameters: str, x_min: float, x_max: float) -> bool:
    pass
if __name__ == "__main__":
    while True:
        user_input = input("What graph do you want to plot? (or say 'bye' to exit): ")
        llm_response = get_llm_response_v2(user_input)
        print(f"LLM response: {llm_response}")
        action, function_type, function_params, x_min, x_max = parse_llm_response_v2(llm_response)

        if action == "exit":
            print("Thank you, goodbye!")
            break
        elif action == "plot":
            if plot_graph_v1(function_name, x_min, x_max):
                print("Graph plotted successfully.")
            else:
                print("Sorry, there was an error plotting the graph.")
        elif action == "unknown":
            print(
                "Sorry, I didn't understand your request or the function is not supported. Please ask again using supported functions: " + allowed_functions_str_v1)
        elif action == "error":
            print("Sorry, there was an error communicating with the LLM. Please try again later.")