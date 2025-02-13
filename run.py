#1. Вычисление энтропии для каждого разбиения
#2. Разбиение по наилучшему разбиению по приросту информации
#3. Рекурсивное построение дерева

import numpy as np
from collections import Counter

def print_tree(node, indent="", feature_names=None):
    if isinstance(node, dict):
        feature_index = node['feature_index']
        threshold = node['threshold']
        feature_name = feature_names[feature_index] if feature_names else f"Feature {feature_index}"
        print(f"{indent}{feature_name} <= {threshold:.2f}")
        print_tree(node['left'], indent + "  ", feature_names)
        print(f"{indent}{feature_name} > {threshold:.2f}")
        print_tree(node['right'], indent + "  ", feature_names)
    else:
        print(f"{indent}Class: {node}")

def entropy(y):
  hist = np.bincount(y)
  ps = hist / len(y)
  return -np.sum([p * np.log2(p) for p in ps if p > 0])

def split(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def calc_child_entropy(X_left, X_right, y_left, y_right, y):
  n = len(y)
  n_left, n_right = len(y_left), len(y_right)
  child_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
  
  return child_entropy

def information_gain(X, y, feature_index, threshold):
  parent_entropy = entropy(y)

  X_left, X_right, y_left, y_right = split(X, y, feature_index, threshold)

  if len(X_left) == 0 or len(X_right) == 0:
    return 0

  child_entropy = calc_child_entropy(X_left, X_right, y_left, y_right, y)

  # Прирост информации
  ig = parent_entropy - child_entropy
  return ig

def best_split(X, y):
    best_feature_index, best_threshold, best_ig = None, None, -1
    
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        
        for threshold in thresholds:
            ig = information_gain(X, y, feature_index, threshold)
            if ig > best_ig:
                best_ig = ig
                best_feature_index = feature_index
                best_threshold = threshold
                
    return best_feature_index, best_threshold

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return leaf_value
        
        feature_index, threshold = best_split(X, y)
        
        if feature_index is None:
            return self._most_common_label(y)
        
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        left_subtree = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth + 1)
        
        return {'feature_index': feature_index, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if isinstance(node, dict):
            feature_index = node['feature_index']
            threshold = node['threshold']

            if x[feature_index] <= threshold:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])
        else:
            return node

#example

# Пример данных
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
y = np.array([0, 0, 1, 1, 1, 0, 0])

# Создание и обучение дерева
tree = DecisionTree(max_depth=10)
tree.tree = tree.fit(X, y)

# Предсказание
predictions = tree.predict(X)
print(predictions)