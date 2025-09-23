# Decision Tree Explained
A **decision tree** is a supervised machine learning algorithm used for classification and regression tasks. It models decisions and their possible consequences as a tree-like structure.

![Decision Tree Animation](https://towardsdatascience.com/wp-content/uploads/2024/01/1ZozskkFCFt6biaMJvNgnhQ.gif)

## How It Works

1. **Splitting:** The algorithm starts at the root node and splits the data into subsets based on feature values.
2. **Decision Nodes:** Each internal node represents a test on a feature.
3. **Leaf Nodes:** Each leaf node represents a predicted outcome or class.
4. **Recursive Partitioning:** The process repeats recursively for each subset, creating branches until a stopping criterion is met (e.g., maximum depth or minimum samples per leaf).

## Example

```
Is age < 30?
├── Yes: Is income > $50K?
│   ├── Yes: Approve loan
│   └── No: Deny loan
└── No: Approve loan
```

## Advantages

- Easy to understand and interpret
- Handles both numerical and categorical data

## Limitations

- Prone to overfitting
- Can be unstable with small data changes

Decision trees are foundational in machine learning and are often used as building blocks for more complex models like Random Forests and Gradient Boosted Trees.