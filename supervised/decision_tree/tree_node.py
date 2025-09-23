import numpy as np

class TreeNode():
    """
    Represents a node in the decision tree.

    Attributes:
        data (np.array): Samples at this node.
        feature_idx (int): Index of feature used for the split.
        feature_val (float): Threshold value for the split.
        prediction_probs (np.array): Label probability distribution.
        information_gain (float): Info gain from the split.
        left (TreeNode): Left child node.
        right (TreeNode): Right child node.
    """
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None

    def node_def(self) -> str:
        """
        Returns a string describing the node.

        - If it has children: shows split rule and info gain.
        - If it’s a leaf: shows label counts and prediction probs.

        Returns:
            str: Node description.
        """
        if (self.left or self.right):
            return f"NODE | Information Gain = {self.information_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
        else:
            unique_values, value_counts = np.unique(self.data[:,-1], return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])
            return f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"
