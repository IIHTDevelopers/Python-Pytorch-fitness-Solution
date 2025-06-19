import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from fitness import FitnessDataset, build_model, train_model, bin_calories, load_data_from_csv, evaluate_model
from tests.TestUtils import TestUtils

class TestFitnessModelYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        # Load data using the correct function from fitness.py
        X_train, y_train, X_test, y_test = load_data_from_csv("fitness_data.csv")
        cls.dataset = FitnessDataset(X_train, y_train)
        cls.dataloader = DataLoader(cls.dataset, batch_size=8, shuffle=False)
        cls.model = build_model(input_size=5, num_classes=3)
        cls.sample_input = X_train[0:1]  # Use preprocessed data

    def test_dataset_length(self):
        try:
            # Load the original data to get the expected length
            X_train, y_train, X_test, y_test = load_data_from_csv("fitness_data.csv")
            result = len(self.dataset) == len(X_train)
            self.test_obj.yakshaAssert("TestDatasetLength", result, "functional")
            print("TestDatasetLength =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestDatasetLength", False, "functional")
            print("TestDatasetLength = Failed | Exception:", e)

    def test_model_output_shape(self):
        try:
            output = self.model(self.sample_input)
            result = output.shape == torch.Size([1, 3])  # 3 classes for classification
            self.test_obj.yakshaAssert("TestModelOutputShape", result, "functional")
            print("TestModelOutputShape =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelOutputShape", False, "functional")
            print("TestModelOutputShape = Failed | Exception:", e)


    def test_prediction_value(self):
        try:
            self.model.eval()
            with torch.no_grad():
                pred = self.model(self.sample_input)
                # For classification, check if we get valid class probabilities/logits
                _, predicted_class = torch.max(pred, 1)
                result = predicted_class.item() in [0, 1, 2]  # Valid class indices
                self.test_obj.yakshaAssert("TestPredictionReasonableValue", result, "functional")
                print("TestPredictionReasonableValue =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictionReasonableValue", False, "functional")
            print("TestPredictionReasonableValue = Failed | Exception:", e)

    def test_model_accuracy(self):
        try:
            # Load data
            X_train, y_train, X_test, y_test = load_data_from_csv("fitness_data.csv")
            train_dataset = FitnessDataset(X_train, y_train)
            test_dataset = FitnessDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=4)
            test_loader = DataLoader(test_dataset, batch_size=4)

            # Build and train model
            model = build_model(input_size=X_train.shape[1])
            train_model(model, train_loader, epochs=15)

            # Evaluate
            accuracy = evaluate_model(model, test_loader)
            result = accuracy >= 0.90

            self.test_obj.yakshaAssert("TestModelAccuracy", result, "functional")
            print("TestModelAccuracy =", "Passed" if result else f"Failed ({accuracy:.2%})")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
            print("TestModelAccuracy = Failed | Exception:", e)

    def test_build_model_structure(self):
        try:
            input_size = 5
            num_classes = 3
            model = build_model(input_size=input_size, num_classes=num_classes)

            # Sample input
            sample_input = torch.randn(1, input_size)
            output = model(sample_input)

            # Check output shape
            correct_output_shape = output.shape == torch.Size([1, num_classes])

            # Check for Dropout layer
            has_dropout = any(isinstance(layer, nn.Dropout) for layer in model)

            result = correct_output_shape and has_dropout
            self.test_obj.yakshaAssert("TestBuildModelStructure", result, "functional")
            print("TestBuildModelStructure =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestBuildModelStructure", False, "functional")
            print("TestBuildModelStructure = Failed | Exception:", e)

    def test_fitness_dataset_behavior(self):
        try:
            # Load real data
            X_train, y_train, _, _ = load_data_from_csv("fitness_data.csv")

            # Create dataset instance
            dataset = FitnessDataset(X_train, y_train)

            # Assertions
            length_check = len(dataset) == len(X_train)
            item_check = isinstance(dataset[0], tuple) and len(dataset[0]) == 2

            # Input and label shape check (optional but useful)
            shape_check = dataset[0][0].shape == X_train[0].shape and isinstance(dataset[0][1].item(), int)

            result = length_check and item_check and shape_check

            self.test_obj.yakshaAssert("TestFitnessDatasetBehavior", result, "functional")
            print("TestFitnessDatasetBehavior =", "Passed" if result else "Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestFitnessDatasetBehavior", False, "functional")
            print("TestFitnessDatasetBehavior = Failed | Exception:", e)

    def test_stratified_split_ratio(self):
        try:
            # Load the data using the actual function
            X_train, y_train, X_test, y_test = load_data_from_csv("fitness_data.csv")

            # Check total size
            total_samples = len(X_train) + len(X_test)
            expected_train_size = int(0.8 * total_samples)
            expected_test_size = total_samples - expected_train_size

            # Allow minor deviation due to integer rounding
            train_ok = abs(len(X_train) - expected_train_size) <= 1
            test_ok = abs(len(X_test) - expected_test_size) <= 1

            result = train_ok and test_ok

            self.test_obj.yakshaAssert("TestStratifiedSplitRatio", result, "functional")
            print("TestStratifiedSplitRatio =", "Passed" if result else "Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestStratifiedSplitRatio", False, "functional")
            print("TestStratifiedSplitRatio = Failed | Exception:", e)

if __name__ == "__main__":
    unittest.main()
