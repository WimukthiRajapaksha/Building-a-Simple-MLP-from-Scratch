# Building a Simple MLP from Scratch

A multilayer perceptron written using only NumPy that classifies the [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) into its three species: *Setosa*, *Versicolor* and *Virginica*. The forward pass, backpropagation and SGD updates are all implemented by hand, with no deep learning framework.

This was written as coursework for an MSc applied AI/ML course. scikit-learn is used only to load, split and standardize the data, and Matplotlib only for plots.

## Results

The model reaches **96.67% test accuracy** (29 of 30 samples) after 100 epochs, with 94.17% training accuracy and a final loss of 0.2020.

| True \ Predicted | Setosa | Versicolor | Virginica |
| --- | ---: | ---: | ---: |
| **Setosa** | 10 | 0 | 0 |
| **Versicolor** | 0 | 9 | 1 |
| **Virginica** | 0 | 0 | 10 |

The only error is a Versicolor flower predicted as Virginica, the two species whose measurements overlap.

Treat the accuracy as an optimistic estimate. The test set is small, so one sample moves the score by 3.3 percentage points, and the random seed was tuned against accuracy rather than chosen in advance.

## Model

```
Input (4 features) → Linear → ReLU (16 units) → Linear → Softmax (3 classes)
```

| Setting | Value |
| --- | --- |
| Loss | Categorical cross-entropy |
| Optimizer | Mini-batch SGD |
| Learning rate | 0.01 |
| Batch size | 16 |
| Epochs | 100 |
| Weight initialization | `N(0, 1) * sqrt(2 / fan_in)` (He) |
| Data split | 80% train, 20% test (120 / 30 samples), stratified |
| Preprocessing | Standardized features, with the scaler fit on training data only |
| Random seed | 2 |

## Repository contents

| File | Description |
| --- | --- |
| [`assignment-1_wimukthi_a1988198.ipynb`](assignment-1_wimukthi_a1988198.ipynb) | Implementation, training run, plots and discussion questions |
| [`assignment-1_wimukthi_a1988198.pdf`](assignment-1_wimukthi_a1988198.pdf) | PDF export of the notebook |

## Running the notebook

```bash
git clone https://github.com/WimukthiRajapaksha/Building-a-Simple-MLP-from-Scratch.git
cd Building-a-Simple-MLP-from-Scratch
pip install numpy scikit-learn matplotlib notebook
jupyter notebook assignment-1_wimukthi_a1988198.ipynb
```

Then run all cells. Training takes a few seconds on a CPU, and the dataset ships with scikit-learn, so there is nothing to download. The run saves two plots to the working directory: `trainig_curve.png` and `confusion_matrix.png`.

The results above reproduce with Python 3.12, NumPy 2.4, scikit-learn 1.8 and Matplotlib 3.10.

## Notebook outline

1. Imports and setup
2. Data loading and preprocessing
3. MLP implementation: activations, loss, forward and backward passes
4. Mini-batch SGD training loop
5. Training run and final accuracy
6. Loss and accuracy curves, and the confusion matrix
7. Discussion questions: why ReLU is used in the hidden layer, and what causes poor test performance
