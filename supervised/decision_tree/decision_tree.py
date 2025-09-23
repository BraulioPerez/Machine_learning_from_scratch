import numpy as np
from collections import Counter
from tree_node import TreeNode


class DecisionTree():
    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, max_depth=4, min_samples_leaf=1, min_information_gain=0.0) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain

    def entropy(self, class_probabilities: list) -> float:
        """
        Calculate the entropy of a dataset, which measures how uncertain or mixed the classes are.

        In a decision tree, entropy tells us how “impure” a set of examples is:
        - Entropy = 0 → all examples belong to the same class (completely pure)
        - Higher entropy → examples are more evenly spread across classes (more uncertainty)

        Args:
            class_probabilities (list): Probabilities of each class in the dataset (values between 0 and 1).

        Returns:
            float: Entropy value representing the level of disorder in the dataset.
                Used to find the best splits in a decision tree.

        Example:
            class_probabilities = [0.5, 0.5]
            entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
            # Maximum uncertainty for two classes
        """
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])

    def class_probabilities(self, labels: list) -> list:
        """
        Calculate the probability of each class in a dataset.

        This method counts how many examples belong to each class and converts it into
        probabilities. These probabilities are used to compute measures like entropy
        when deciding how to split a dataset in a decision tree.

        Args:
            labels (list): List of class labels for the examples in the dataset.

        Returns:
            list: Probabilities of each class (values between 0 and 1), summing to 1.

        Example:
            labels = ['A', 'A', 'B', 'B', 'B']
            class_probabilities = [2/5, 3/5]  # 'A' occurs 2 times, 'B' occurs 3 times
        """
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_entropy(self, labels: list) -> float:
        """
        Calculate the entropy of a dataset based on its labels.

        This method first computes the probability of each class in the dataset
        and then calculates the entropy. Entropy measures how mixed or uncertain
        the dataset is, which is useful for decision trees to determine the best splits.

        Args:
            labels (list): List of class labels for the examples in the dataset.

        Returns:
            float: Entropy value representing the level of disorder or uncertainty
                in the dataset.

        Example:
            labels = ['A', 'A', 'B', 'B', 'B']
            probabilities = [2/5, 3/5]
            entropy = -2/5*log2(2/5) - 3/5*log2(3/5) ≈ 0.971
        """
        return self.entropy(self.class_probabilities(labels))

    def partition_entropy(self, subsets: list) -> float:
        """
        Calculate the overall entropy after splitting a dataset into subsets.

        This method is used in decision trees to measure how "uncertain" the
        data is after making a split. Each subset has its own entropy, and the
        function takes a weighted average of these entropies based on the size
        of each subset.

        Args:
            subsets (list): A list of lists, where each inner list contains
                the labels of one subset.
                Example: [[1, 0, 0], [1, 1, 1], [0, 0, 1, 0, 0]]

        Returns:
            float: The weighted entropy of all subsets combined.
                A lower value means the split creates purer groups,
                which is better for a decision tree.

        Example:
            subsets = [[1, 0], [1, 1, 1]]
            # First subset entropy ≈ 1.0 (mixed 50/50)
            # Second subset entropy = 0 (all the same)
            partition_entropy = (2/5 * 1.0) + (3/5 * 0) = 0.4
        """
        total_count = sum([len(subset) for subset in subsets])
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])

    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        """
        Split a dataset into two groups based on a feature threshold.

        This method checks the values of one feature (a column in the dataset)
        and divides the data into two groups:
        - Group 1: rows where the feature value is less than the threshold
        - Group 2: rows where the feature value is greater or equal

        This step is used in decision trees to create branches by
        splitting the data into smaller, more "pure" subsets.

        Args:
            data (np.array): The dataset represented as a 2D NumPy array,
                where rows are examples and columns are features.
            feature_idx (int): The index (column number) of the feature
                used for the split.
            feature_val (float): The threshold value to compare the feature against.

        Returns:
            tuple: Two NumPy arrays (group1, group2) representing the
                subsets of the data after the split.

        Example:
            data = np.array([
                [2.5, 'A'],
                [1.0, 'B'],
                [3.2, 'A']
            ])
            group1, group2 = split(data, feature_idx=0, feature_val=2.0)
            # group1 will contain rows with feature < 2.0 → [[1.0, 'B']]
            # group2 will contain rows with feature >= 2.0 → [[2.5, 'A'], [3.2, 'A']]
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2

    def find_best_split(self, data: np.array) -> tuple:
        """
        Find the best way to split a dataset by minimizing entropy.

        This method looks at each feature (column) in the dataset and tries
        splitting the data at the median value of that feature. For each split,
        it calculates the partition entropy (how mixed the labels are after
        the split). The function then chooses the feature and threshold that
        produce the lowest entropy, meaning the groups are as "pure" as possible.

        Args:
            data (np.array): A 2D NumPy array where:
                - Rows are examples
                - Columns are features, and the last column is the class label

        Returns:
            tuple:
                - g1_min (np.array): First group of data (feature < best threshold)
                - g2_min (np.array): Second group of data (feature >= best threshold)
                - min_entropy_feature_idx (int): Index of the feature chosen for the split
                - min_entropy_feature_val (float): Threshold value used for the split
                - min_part_entropy (float): The entropy value of the best split

        Example:
            data = np.array([
                [2.5, 1],
                [1.0, 0],
                [3.2, 1],
                [2.0, 0]
            ])
            g1, g2, feature_idx, feature_val, entropy = find_best_split(data)

            # Suppose the best split is on column 0 at value 2.25
            # g1 contains rows with feature < 2.25
            # g2 contains rows with feature >= 2.25
            # feature_idx = 0
            # feature_val = 2.25
            # entropy is the partition entropy of this split
        """
        min_part_entropy = 1e6
        min_entropy_feature_idx = None
        min_entropy_feature_val = None

        for idx in range(data.shape[1]-1):
            feature_val = np.median(data[:, idx])
            g1, g2 = self.split(data, idx, feature_val)
            part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]])
            if part_entropy < min_part_entropy:
                min_part_entropy = part_entropy
                min_entropy_feature_idx = idx
                min_entropy_feature_val = feature_val
                g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def find_label_probs(self, data: np.array) -> np.array:
        """
        Calculate the probability of each label in a dataset.

        This method looks at the class labels in the last column of the dataset
        and computes how often each label appears. The result is a probability
        distribution, showing the chance of picking each label at random.

        This is useful in decision trees for measuring impurity
        (e.g., entropy or information gain).

        Args:
            data (np.array): A 2D NumPy array where:
                - Rows are examples
                - Columns are features, and the last column is the class label

        Returns:
            np.array: An array of probabilities, one for each label
                (values between 0 and 1, summing to 1).

        Example:
            Suppose we have 5 examples with labels in the last column:
            data = np.array([
                [2.5, 1],
                [1.0, 0],
                [3.2, 1],
                [2.0, 0],
                [1.5, 0]
            ])

            If labels_in_train = [0, 1], then:
            - Label 0 appears 3 times → probability = 3/5 = 0.6
            - Label 1 appears 2 times → probability = 2/5 = 0.4

            find_label_probs(data) → [0.6, 0.4]
        """
        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursively build a decision tree from the dataset.

        This method creates a tree node at each step by finding the best feature
        to split the data, calculating label probabilities, and computing information gain.
        It stops splitting when certain conditions are met, such as reaching the
        maximum depth, having too few samples in a leaf, or if the information gain is too small.

        Args:
            data (np.array): A 2D NumPy array where:
                - Rows are examples
                - Columns are features, with the last column as the class label
            current_depth (int): The current depth of the tree (starts at 0).

        Returns:
            TreeNode: A node in the decision tree containing:
                - data: the subset of data at this node
                - split_feature_idx: the index of the feature used to split
                - split_feature_val: the threshold used for the split
                - label_probabilities: probability of each class at this node
                - information_gain: the gain obtained by the split
            The node may have left and right children if further splitting is possible.

        Example:
            Suppose we have a dataset with 5 examples and 2 features:
            data = np.array([
                [2.5, 1, 0],
                [1.0, 0, 1],
                [3.2, 1, 0],
                [2.0, 0, 1],
                [1.5, 1, 0]
            ])
            tree = create_tree(data, current_depth=0)

            # The method will:
            # 1. Find the best feature and threshold to split the data
            # 2. Calculate label probabilities at this node
            # 3. Compute information gain
            # 4. Create a TreeNode
            # 5. Recursively repeat steps 1-4 for left and right splits
            # until stopping criteria are reached (max depth, min samples, or low info gain)
        """
        # Check if the max depth has been reached (stopping criteria)
        if current_depth >= self.max_depth:
            return None

        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)

        # Find label probs for the node
        label_probabilities = self.find_label_probs(data)

        # Calculate information gain
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - split_entropy

        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)

        return node

    def predict_one_sample(self, X: np.array) -> np.array:
        """
        Predict the class probabilities for a single example.

        This method takes one example (a 1D array of feature values) and
        traverses the decision tree from the root to a leaf. At each node,
        it checks the feature value against the split threshold and moves
        left or right accordingly. Once it reaches a leaf, it returns the
        predicted probabilities of each class.

        Args:
            X (np.array): A 1D NumPy array representing a single example's features.

        Returns:
            np.array: Probabilities of each class for this example.
                Values are between 0 and 1 and sum to 1.

        Example:
            Suppose we have a decision tree for 2 classes (0 and 1):
            X = np.array([2.0, 1.5])
            prediction = predict_one_sample(X)
            # prediction might be [0.6, 0.4], meaning class 0 probability = 0.6,
            # class 1 probability = 0.4
        """
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Train the decision tree on a dataset.

        This method combines the features and labels into one array,
        records the unique labels in the training set, and then
        starts building the tree recursively.

        Args:
            X_train (np.array): 2D array of training features (rows = examples, columns = features)
            Y_train (np.array): 1D array of labels corresponding to X_train

        Returns:
            None

        Example:
            X_train = np.array([[2.5, 1], [1.0, 0], [3.2, 1]])
            Y_train = np.array([0, 1, 0])
            train(X_train, Y_train)
            # The tree is now built and stored in self.tree
        """
        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Start creating the tree
        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict_proba(self, X_set: np.array) -> np.array:
        """
        Predict class probabilities for multiple examples.

        This method applies the single-sample prediction function
        to every row in the dataset, returning probabilities for all examples.

        Args:
            X_set (np.array): 2D array of feature data (rows = examples, columns = features)

        Returns:
            np.array: Array of predicted probabilities for each example.
                Each row sums to 1 and shows the probability for each class.

        Example:
            X_set = np.array([[2.0, 1.5], [1.0, 0.5]])
            predict_proba(X_set)
            # Might return [[0.6, 0.4], [0.3, 0.7]]
        """

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)

        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Predict the class labels for multiple examples.

        This method uses predict_proba to get probabilities for all examples,
        and then picks the class with the highest probability for each example.

        Args:
            X_set (np.array): 2D array of feature data (rows = examples, columns = features)

        Returns:
            np.array: Predicted class labels (integers) for each example.

        Example:
            X_set = np.array([[2.0, 1.5], [1.0, 0.5]])
            predict(X_set)
            # Might return [0, 1]
        """

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds

    def print_recursive(self, node: TreeNode, level=0) -> None:
        """
        Recursively print the decision tree structure.

        This method prints each node of the tree in an indented format,
        showing the splits and node definitions. Left children are printed
        first, then the node, then right children.

        Args:
            node (TreeNode): The current node to print
            level (int): Current depth level used for indentation (default = 0)

        Returns:
            None

        Example:
            print_recursive(tree.root)
            # Prints the tree starting from the root node
        """
        if node != None:
            self.print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self.print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        """
        Print the entire decision tree from the root.

        This method starts the recursive printing process from the root node.

        Args:
            None

        Returns:
            None

        Example:
            print_tree()
            # Prints all nodes of the tree in a readable format
        """
        self.print_recursive(node=self.tree)