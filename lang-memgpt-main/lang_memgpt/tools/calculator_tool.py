from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing import Any, List, Optional


@tool
async def calculate(
    expression: str,  # Matches docstring
    # Added optional config for framework compatibility
    config: Optional[RunnableConfig] = None
) -> Optional[float]:
    """
    Evaluate a basic mathematical expression with +, -, *, /.

    Args:
        expression (str): The mathematical expression to evaluate.
        config (Optional[RunnableConfig]): Optional runtime configuration.

    Returns:
        Optional[float]: The result of the calculation, or None if an error occurs.

    Example Usage:
        calculate("3 + 5 * 2 - 4 / 2")

    Raises:
        ValueError: If the expression is invalid or contains unsupported operations.
    """
    import re
    import operator

    try:
        # Supported operators
        operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }

        # Tokenize expression (numbers and operators)
        tokens = re.findall(r'\d+\.?\d*|[+\-*/]', expression.replace(' ', ''))

        # Parse tokens into valid numbers or operators
        def parse_token(token):
            try:
                return float(token)  # Convert numbers
            except ValueError:
                if token in operators:
                    return token  # Valid operator
                raise ValueError(f"Unsupported token: {token}")

        # Process tokens
        tokens = [parse_token(token) for token in tokens]

        # Evaluate the expression (left-to-right)
        result = tokens[0]  # Start with the first number
        i = 1
        while i < len(tokens):
            operator = tokens[i]
            number = tokens[i + 1]

            # Handle division by zero
            if operator == '/' and number == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")

            # Perform operation
            result = operators[operator](result, number)
            i += 2  # Move to next operator and number

        return result

    except ZeroDivisionError as e:
        # Handle division by zero
        return f"Error: {e}"

    except Exception as e:
        # Handle general errors
        return f"Error evaluating expression: {e}"
