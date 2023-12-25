import re

def remove_docstrings(code: str) -> str:
    """
    Remove docstrings from code snippet
    """
    docstring_pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'
    cleaned_code = re.sub(docstring_pattern, '', code, flags=re.DOTALL)
    return cleaned_code
    
def replace_class_and_function_names(code: str) -> str:
    """
    Replace all class and function names in the given code snippet with 'cls' and 'func' respectively.
    """
    # Regular expression to find class and function definitions
    class_pattern = r"\bclass\s+(\w+)"
    function_pattern = r"\bdef\s+(\w+)"

    # Replace class names with 'cls'
    replaced_code = re.sub(class_pattern, "class cls", code)

    # Replace function names with 'func', excluding special methods like __init__
    replaced_code = re.sub(function_pattern, lambda m: "def func" if not m.group(1).startswith('__') else m.group(0), replaced_code)

    return replaced_code


def extract_code_elements(code: str) -> dict:
    """
    Extract class names, function names, class fields, and function arguments from the given code snippet.
    """
    # Regular expressions for class and function definitions, class fields, and function arguments
    class_pattern = r"\bclass\s+(\w+)"
    function_pattern = r"\bdef\s+(\w+)"
    class_field_pattern = r"\bself\.(\w+)"
    function_arg_pattern = r"\bdef\s+\w+\(([^)]*)\)"
    docstring_pattern = r'""".*?"""|\'\'\'.*?\'\'\''

    # Extract class and function names
    class_names = re.findall(class_pattern, code)
    function_names = re.findall(function_pattern, code)

    # Extract class fields and function arguments
    class_fields = re.findall(class_field_pattern, code)
    function_args = re.findall(function_arg_pattern, code)
    
    # Extract docstrings 
    docstrings = re.findall(docstring_pattern, code, re.DOTALL)

    # Process function arguments to split them into individual arguments
    processed_function_args = []
    for args in function_args:
        args = args.replace(' ', '').split(',')
        # Remove 'self' from arguments
        args = [arg for arg in args if arg != 'self' and arg]
        args = [arg.split('=')[0].strip() for arg in args]
        processed_function_args.extend(args)

    # Create a dictionary with unique names
    unique_names = {
        "class_names": list(set(class_names)),
        "function_names": list(set(function_names) - set(['__init__', '__str__', '__len__'])),
        "class_fields": list(set(class_fields)),
        "function_args": list(set(processed_function_args)),
        "docstrings": docstrings,
    }

    return unique_names