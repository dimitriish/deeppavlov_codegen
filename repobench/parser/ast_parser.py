import ast

class CodeAnalyzer(ast.NodeVisitor):
    """collect code elements via ast parsing"""
    def __init__(self):
        self.defined_classes = []
        self.called_classes = []
        self.defined_functions = []
        self.called_names = []
        self.class_attributes = []
        self.function_arguments = []
        self.docstrings = []

    def visit_ClassDef(self, node):
        self.defined_classes.append(node.name)
        docstring = ast.get_docstring(node)
        if docstring:
            self.docstrings.append(docstring)
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                self.class_attributes.append(target.attr)
            self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.docstrings.append(ast.get_docstring(node))
        args = [arg.arg for arg in node.args.args if arg.arg != 'self']
        
        if node.name == '__init__':
            self.class_attributes.extend(args)
        elif node.name.startswith('__'):
            self.function_arguments.extend(args)

        self.defined_functions.append(node.name)
        self.function_arguments.extend(args)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.called_names.append(node.func.attr)
            if hasattr(node.func.value, 'id') and node.func.value.id in self.defined_classes:
                self.called_classes.append(node.func.value.id)
        elif isinstance(node.func, ast.Name):
            self.called_names.append(node.func.id)
        self.generic_visit(node)


def extract_code_elements_ast(code):
    tree = ast.parse(code)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    return {
        "defined_classes": list(set(analyzer.defined_classes)- set([''])),
        "called_names": list(set(analyzer.called_names) - set([''])),
        "defined_functions": list(set([f for f in analyzer.defined_functions if not f.startswith('__')])),
        "class_attributes": list(set(analyzer.class_attributes)- set([''])),
        "function_arguments": list(set(analyzer.function_arguments)- set([''])),
        "docstrings": list(set([d for d in analyzer.docstrings if d]))
    }

def prepare_code_for_ast(code):
    lines = code.split('\n')

    for i in range(len(lines)):
        if lines[i].strip().endswith(':'):
            if i == len(lines) - 1:
                current_indentation = len(lines[i]) - len(lines[i].lstrip())
                lines[i] += '\n' + ' ' * (current_indentation + 4) + 'pass'
            else:
                current_indentation = len(lines[i]) - len(lines[i].lstrip())
                next_indentation = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                if next_indentation <= current_indentation:
                    lines[i] += '\n' + ' ' * (current_indentation + 4) + 'pass'
    prepared_code = '\n'.join(lines)
    return prepared_code