from .Functions_24_DiGregorio_Castaldo import GaussianSVMComplete
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from typing import Optional, Dict


class MulticlassSVM:
    """
    Class implementing a one-versus-all gaussian kernel Support Vector Machine for a three classes problem
    (easily extendable to any number of classes). It simply relies on the SVM specification already implemented
    in other modules.
    """

    def __init__(self, gamma, inv_reg) -> None:
        """
        Initialization of a Support Vector classifier for three classes problems.
        :param gamma: The gamma hyperparameter for the Gaussian kernel.
        :param inv_reg: The inverse regularization term, a linear scaling on the hinge loss of the binary problem.
        """
        # Initialize list of SVM model objects
        self.models: list[GaussianSVMComplete, ...] = [
            GaussianSVMComplete(inv_reg, gamma) for x in range(3)
        ]
        self.encoder: Optional[
            LabelBinarizer
        ] = None  # Encoder object for labels, initialized later on

    def fit(self, train_data: np.ndarray, tol: float, max_iter: int) -> Dict:
        """
        Fitting method. Three different gaussian SVM models are fitted for the three-classes problem, in a
        one-versus-all fashion.
        :param train_data: The training data, last column is expected to be the response one.
        :param tol: Tolerance for SMO algorithm optimality condition.
        :param max_iter: Maximum number of iterations for SMO algorithm.
        :return: A dictionary with information on the optimization process.
        """
        # One hot encoder object of the response variable
        # Negative value is -1, so it is already in the right format
        self.encoder = LabelBinarizer(neg_label=-1).fit(train_data[:, -1])
        # Transform response vector in encoded response array
        response = self.encoder.transform(train_data[:, -1])
        train_data = np.concatenate([train_data[:, :-1], response], axis=1)
        opt_status = list()
        # Fit three different models, each on a different column of the encoded response array (implicitly one vs. all)
        for i, model in enumerate(self.models):
            opt_status.append(
                model.smo_fit(
                    train_data[:, :-3],
                    train_data[:, -3 + i],
                    tol=tol,
                    max_iter=max_iter,
                )
            )

        tot_time = 0
        difference = 0
        tot_iter = 0
        for i in range(3):
            tot_time += opt_status[i]["Time"]
            difference += opt_status[i]["KKTViolation"]
            tot_iter += opt_status[i]["Iterations"]
        return {
            "TotalTime": tot_time,
            "AverageDifference": difference / 3,
            "TotalIter": tot_iter,
        }

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Prediction method for the three-class Gaussian SVM.
        :data: The data to predict on, last column is expected to hold the response variable.
        :return: A vector of decoded predictions.
        """
        pred_list = []
        for i, model in enumerate(self.models):  # Get predictions from each model
            pred_list.append(model.predict(data, score=True))  # Get score returned
        preds = np.column_stack(pred_list)  # Column stacking of predictions
        # We always get the class having the higher score, this covers all the situations we can face
        preds = np.argmax(preds, axis=1)

        return np.array(
            [self.encoder.classes_[x] for x in preds]
        )  # Decode the response and return
