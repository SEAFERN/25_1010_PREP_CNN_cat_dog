# Cat/Dog Classifier with CNN

This project will include:
- `main.py`: Main script for training and testing a simple CNN classifier.
- Instructions for downloading a small cat/dog dataset.
- Beginner-friendly explanations in the README.

Recommended structure:
- `main.py` (CNN code)
- `README.md` (instructions)
- `data/` (for images, not included by default)

---

## Next Steps
1. Implement the main Python script for the CNN model.
2. Add dataset download instructions.
3. Update README with setup and usage details.

---

## Dataset Instructions

To train and test the classifier, you need a dataset of cat and dog images. A popular option is the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

**Steps to prepare the data:**
1. Download the dataset from Kaggle (requires a Kaggle account).
2. Unzip the dataset.
3. Organize the images into the following folders:
   - `data/train/cats/` (cat images)
   - `data/train/dogs/` (dog images)
   - `data/validation/cats/` (cat images for validation)
   - `data/validation/dogs/` (dog images for validation)

You can use a small subset (e.g., 1000 images per class for training, 400 per class for validation) to keep it beginner-friendly and fast.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install tensorflow numpy
   ```
2. **Prepare the dataset:**
   Follow the instructions above to download and organize the images.
3. **Train the model:**
   ```bash
   python main.py
   ```
   The script will train a simple CNN and save the model as `cat_dog_cnn_model.h5`.

## Understanding the Code
- The code uses TensorFlow/Keras to build a simple CNN for binary classification (cat vs. dog).
- Images are loaded from folders using Keras' `ImageDataGenerator`.
- The model is trained for 5 epochs (you can change this in `main.py`).
- After training, the model is saved for future use.

Feel free to experiment with the code and try different architectures or more data!