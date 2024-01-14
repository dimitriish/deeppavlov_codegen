"""
# Symmetric and Asymmetric Code Retrieval Project

## Models Overview

### Initial Models
1. **Random Model**: A baseline model using random selection.
2. **Jaccard Similarity**: Compares similarity between code snippets based on shared tokens.
3. **Edit Similarity**: Measures similarity by the number of edits required to transform one snippet into another.
4. **UnixCoder**: A language model specifically trained for coding tasks.

### Advanced Models
- **BM25**: A text retrieval model that ranks documents based on the frequency of query terms.
- **Sentence Transformers**: Utilizes transformer models for generating sentence embeddings.
- **MTEB Models**:
   - **e5-large**: A large-scale multi-task language model.
   - **llmrails/ember-v1**: A specialized model for code-related tasks.

## Experiments

### Dataset Utilization
- **RepoBench**: Used for evaluating repository-level code auto-completion.
We used data from RepoBench - dataset for retrieving and generating code in multi-file projects [Repobench repository](https://github.com/Leolty/repobench)

## Download repobench data
Firstly, clone repo:
```bash
    git clone deeppavlov_codegen
    cd deeppavlov_codegen
```

You can download test data from [Google Drive](https://drive.google.com/file/d/1HvFFnOybTKEJCrEypWh4ftmW6DZBaiK_/view?usp=sharing), or simply run the following command:
   ```bash
    gdown --id '1HvFFnOybTKEJCrEypWh4ftmW6DZBaiK_' --output ./repobench/test.zip
    unzip ./repobench/test.zip -d ./repobench/
    rm ./repobench/test.zip
```

If you also want to download the training data, it can be found [here](https://drive.google.com/file/d/179TXJBfMMbP9FDC_hsdpGLQPmN6iB4vY/view?usp=sharing). Similarly, you can run the following command:
   ```bash
    gdown --id '179TXJBfMMbP9FDC_hsdpGLQPmN6iB4vY' --output ./repobench/train.zip
    unzip ./repobench/train.zip -d ./repobench/
    rm ./repobench/train.zip
```

1. Import the `load_data` function:
   ```python
   from repobench.utils import load_data
   ```

2. Call the function with the desired parameters:
   ```python
   data = load_data(split, task, language, settings, length)
   ```

**Parameters**:
- `split`: Specify whether you want the `train` or `test` split. 
- `task`: Choose between `retrieval`, `completion` and `pipeline`.
- `language`: Select the programming language, either `python` or `java`.
- `settings`: Choose between `cross_file_first`, `cross_file_random`, or `in_file`. You can also provide a list combining these settings.
- `length`: (Optional) For the `completion` task, please specify the length as either `2k` or `8k`.

**Return**:
- If a single setting is provided, the function returns the loaded data for that setting.
- If multiple settings are specified, the function returns a list containing data for each setting.

For example:
```python
data = load_data(split='test', task='retrieval', language='python', settings=['cross_file_first'])
```
![image](https://github.com/dimitriish/deeppavlov_codegen/assets/62793986/076516d4-7d4e-440c-ac35-caa98591a5c6)


### Code Augmentation Techniques
1. Selecting the last N lines from code snippets.
2. Removing docstrings for clearer code representation.
3. Renaming class and function names to test model robustness.
4. Removing named arguments from the subsequent line of code.
![image](https://github.com/dimitriish/deeppavlov_codegen/assets/62793986/6198cea7-7810-4d7b-94ba-b4869c449f91)

### Task-Specific Approaches
- **Asymmetric Search**: Retrieving the 'gold snippet' using preceding code.
- **Symmetric Search**: Identifying the gold snippet via subsequent code analysis.

## Results

### Performance Metrics
Original paper metrics
![image](https://github.com/dimitriish/deeppavlov_codegen/assets/62793986/5bae85b8-8737-42e9-a4bb-e187bf9f6753)
- **Asymmetric Retrieval**: Achieved state-of-the-art performance on RepoBench's asymmetric retrieval task.
![image](https://github.com/dimitriish/deeppavlov_codegen/assets/62793986/632f3b98-b4ee-42fa-b2d0-2ef63d905ec9)
- **Symmetric Retrieval**: Demonstrated nearly fair performance on symmetric search.
![image](https://github.com/dimitriish/deeppavlov_codegen/assets/62793986/149796b3-e98c-46e9-9263-545204b933ea)

### Impact of Code Corruption
- Analysis on how various code corruption techniques affected the performance of dense and sparse models.
- Insights on the optimal balance between code integrity and augmentation for effective model training.

### Comparative Analysis
- Comparative performance of various models in different retrieval tasks.
- Evaluation of model robustness in the face of code augmentation and transformation.

## Future Directions

- Investigating next line input corruption and time-based code augmentation.
- Exploring the impact of the size of previous lines on model performance.
- Developing methods for explainable code search.
- Enhancing retrieval augmented generation for more sophisticated code auto-completion.
"""
