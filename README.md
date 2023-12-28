# deeppavlov_codegen

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

## Baseline

## Experiments
### Symmetric search
### Asymmetric search
