import argparse
import numpy as np
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn, append_bias_term
import os
import matplotlib.pyplot as plt

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Args:
        args (Namespace): Arguments parsed from the command line.
    """
    ## Load data
    if args.data_type == "features":
        feature_data = np.load('features.npz', allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest = feature_data['xtrain'], feature_data['xtest'],\
                                                      feature_data['ytrain'], feature_data['ytest'],\
                                                      feature_data['ctrain'], feature_data['ctest']
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, 'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ## Prepare data
    if not args.test:
        N = xtrain.shape[0]
        split_index = 2 * N // 3  # Take a third of training data for validation set
        xtest = xtrain[split_index:, :]
        xtrain = xtrain[:split_index, :]
        ytest = ytrain[split_index:]
        ytrain = ytrain[:split_index]
        ctest = ctrain[split_index:]
        ctrain = ctrain[:split_index]
    
    # Normalize
    means = np.mean(xtrain, axis=0, keepdims=True)
    stds = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    ## Initialize the method
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "knn":
        if args.task == "breed_identifying":
            method_obj = KNN(args.K, "classification")
        else:  # center_locating
            method_obj = KNN(args.K, "regression")
    elif args.method == "linear_regression":   
        method_obj = LinearRegression(lmda=args.lmda)
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)
    elif args.method == "logistic_regression":  
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)



    ## Train and evaluate the method
    if args.task == "center_locating":
        preds_train = method_obj.fit(xtrain, ctrain)
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)

        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

    elif args.task == "breed_identifying":
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)

        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    args = parser.parse_args()
    main(args)