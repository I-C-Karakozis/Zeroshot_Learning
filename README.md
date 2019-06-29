# Zero-shot Learning

This repository hosts a pytorch project that performs Zero-shot Animal Classification on the "Animals with Attributes" dataset. For details and references, refer to the PDF report found at the top directory level of the repository.

## Environmental Setup

This project requires Python 3.6.7. To install the python dependencies, run the following command (we recommend using a virtual environment that supports Python 3.6.7):

```bash
pip install -r requirements.txt
```

##  Dataset Download

The dataset annotations can be found in the `annotations` directory. To download the dataset, run the setup script from the top directory level of the repository as follows:

```bash
chmod +x setup.sh
./setup.sh
```

This script will also create all the directories that are needed to train and test your models.

##  Model Training

It is recommended that you train your models on a machine with access to GPU resources. To train your models, run the following python script:

```bash
python main.py
```

To inspect the parameters you can configure for training, execute `python main.py --help`. The `debug` flag should only be set when you want to debug your pipeline.

For more details on model training, refer to the project report.

##  Model Evaluation

It is recommended that you evaluate your models on a machine with access to GPU resources. To evaluate your models, run the following python script:

```bash
python main.py --model {0}
```

`{0}` should be replaced by the name of the file in the `models` directory that stores the weights of the pytorch model you want to evaluate. Models are automatically saved during training.

For more details on model evaluation, refer to the project report.

## Note from the Author
Feedback and comments are highly appreciated. Feel free to open an issue on the repository page to express your thoughts and suggest changes.
