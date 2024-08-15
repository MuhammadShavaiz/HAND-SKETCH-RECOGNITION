## Hand-Drawn Sketch Classification

### Overview

This project tackles the challenge of recognizing hand-drawn sketches. Various models were explored, including ResNet18, ResNet50, and Inception_v3, with Inception_v3 proving to be the most effective, achieving a validation accuracy of 57%.

### Pipeline

The pipeline is constructed using PyTorch and encompasses:

- **Dataset Handling:** A custom dataset class for loading images and labels from a CSV file.
- **Model Training:** Fine-tuning of Inception_v3 with batch processing.
- **Evaluation:** A script to assess model performance and generate predictions.

### File Descriptions

- `dataset.py`: Manages dataset loading and iterator creation.
- `model.py`: Loads and modifies Inception_v3, including parameter counting methods.
- `train.py`: Trains the model for 30 epochs, logs progress, and saves checkpoints.
- `evaluate.py`: Evaluates the dataset using the best model and outputs predictions in CSV format.

### Usage

#### Training

To train the model, ensure your dataset is organized as follows:

```
BaseDir
├── images
│   ├── 1.png
│   ├── 2.png
└── labels.csv
```

Run the training script with the command:

```bash
python3 train.py
```

This script trains the model for 30 epochs, logging progress and saving checkpoints along the way.

#### Evaluation

To evaluate the model:

1. Download and extract the project.
2. Organize the dataset as described above.
3. Run the evaluation script with the command:

```bash
python3 evaluate.py pathToDataset
```

This generates an `evaluation.csv` file with predictions for each image in the dataset.

### Contact

For any questions, please reach out to:

- Muhammad Shavaiz Butt - [shavaizsohail@gmail.com](mailto:shavaizsohail@gmail.com)
- Hasib Aslam - [haslam.bscs21seecs@seecs.edu.pk](mailto:haslam.bscs21seecs@seecs.edu.pk)
