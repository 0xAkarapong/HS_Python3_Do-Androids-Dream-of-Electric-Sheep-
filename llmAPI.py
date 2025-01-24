import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def get_llm_response_v1(user_message: str) -> str:
    instruction = f"""You are a helpful assistant that extracts function and interval for graph plotting.
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
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{instruction}",
            },
        ]
    )
    return chat_response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("What graph do you want to plot? (or say 'bye' to exit): ")
        llm_response = get_llm_response_v1(user_input)
        print(llm_response)