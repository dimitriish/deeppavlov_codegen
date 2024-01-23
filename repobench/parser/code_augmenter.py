import re

class CodeAugmenter():
    def __init__(self, code: str, code_elements: dict):
        self.code = code
        self.code_elements = code_elements

    def remove_comments(self, code: str) -> str:
        return re.sub(r'#.*', '', code)

    
    def remove_docstrings(self, code: str) -> str:
        def replacer(match):
            start, end = match.span()
            return ' ' * (end - start)
            
        pattern = r'""".*?"""|\'\'\'.*?\'\'\''
        return re.sub(pattern, replacer, code, flags=re.DOTALL)

    def rename_entity_names(self, code: str, prefix: str, key: str) -> str:
        modified_code = code
        for i, name in enumerate(self.code_elements[key]):
            new_name = f'{prefix}_{i}'
            modified_code = re.sub(fr'\b{name}\b', new_name, modified_code)
            # modified_code = re.sub(fr'\b{name}\b(?!\', new_name, modified_code)
        return modified_code

    def get_processed_code(self):
        code = self.code
        code = self.remove_comments(code)
        code = self.remove_docstrings(code)
        code = self.rename_entity_names(code, 'cls', 'defined_classes')
        code = self.rename_entity_names(code, 'func', 'defined_functions')
        code = self.rename_entity_names(code, 'var', 'function_arguments')
        code = self.rename_entity_names(code, 'attrib', 'class_attributes')
        return code