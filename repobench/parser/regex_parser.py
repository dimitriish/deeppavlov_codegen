import re

class CodeAnalyzerRegex:
    """collect code elements via ast parsing"""
    def __init__(self, code):
        self.code = code
        self.defined_classes = []
        self.called_names = []
        self.defined_functions = []
        self.class_attributes = []
        self.function_arguments = []
        self.docstrings = []

    def parse(self):
        class_pattern = r"\bclass\s+(\w+)"
        function_pattern = r"\bdef\s+(\w+)\s*\(([^)]*)\)"
        attribute_pattern = r"self\.(\w+)\s*="
        call_pattern = r"\b(\w+)\s*\("
        docstring_pattern = r'("""|\'\'\')([\s\S]*?)\1'

        self.defined_classes = re.findall(class_pattern, self.code)
        functions = re.finditer(function_pattern, self.code)
        for func in functions:
            name, args = func.groups()
            self.defined_functions.append(name)
            if args:
                self.function_arguments.extend([arg.strip() for arg in args.split(',') if arg.strip() != 'self'])

        self.class_attributes = re.findall(attribute_pattern, self.code)
        self.called_names = re.findall(call_pattern, self.code)
        self.docstrings = [match[1] for match in re.findall(docstring_pattern, self.code)]

def extract_code_elements_regex(code: str) -> dict:
    analyzer = CodeAnalyzerRegex(code)
    analyzer.parse()
    return {
        "defined_classes": list(set(analyzer.defined_classes)),
        "called_names": list(set(analyzer.called_names)),
        "defined_functions": list(set([f for f in analyzer.defined_functions if not f.startswith('__')])),
        "class_attributes": list(set(analyzer.class_attributes)),
        "function_arguments": list(set(analyzer.function_arguments) - set(['_', '*args', '**kwargs'])),
        "docstrings": analyzer.docstrings
    }

def remove_docstrings(code: str) -> str:
    """
    Remove docstrings from code snippet
    """
    docstring_pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'
    cleaned_code = re.sub(docstring_pattern, '', code, flags=re.DOTALL)
    return cleaned_code

def modify_arguments_in_code(code):
    code = re.sub(r'\b(\w+)=(\1)\b', r'\2', code)
    code = re.sub(r'\b(\w+)=([^\s,)]+)', r'\2', code)
    return code