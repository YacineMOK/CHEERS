import pickle
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import numpy as np
from typing import List
import os


class Loader:
    def __init__(self, path: str):

        self.path = path

        # Load all previously saved datasets and mappings
        with open(f'{self.path}df_item_idx.pkl', 'rb') as f:
            self.df_item_idx = pickle.load(f)

        with open(f'{self.path}user_item_matrix.pkl', 'rb') as f:
            self.user_item_matrix = pickle.load(f)

        with open(f'{self.path}entity_mappings.pkl', 'rb') as f:
            self.mappings = pickle.load(f)

        with open(f'{self.path}dict_ui.pkl', 'rb') as f:
            self.dict_ui = pickle.load(f)

        self._df_index_preprocessing()
        self.n_users = self.user_item_matrix.shape[0]
        self.n_items = self.user_item_matrix.shape[1]
        self.sparsity = self._compute_sparsity()

        print("Successfully loaded:")
        print(f"- df_item_idx: {self.df_item_idx.shape}")
        print(f"- user_item_matrix: {self.user_item_matrix.shape}")
        print(f"- number of attributes: {len(self.mappings.keys()) - 1}")  # Exclude 'item_to_idx' from count
        print(f"- sparsity: {self.sparsity:.4%} ({self.user_item_matrix.nnz} observed / {self.n_users * self.n_items} possible interactions)")
        print(f"-------------------------\n-------------------------\n")

        self.train_coo = self.user_item_matrix

    def _compute_sparsity(self) -> float:
        """Fraction of (user, item) pairs with no observed interaction."""
        n_possible = self.n_users * self.n_items
        return 1.0 - (self.user_item_matrix.nnz / n_possible)

    def set_training_features(self, features: List[str]):
        self.train_features = features
        self.vocab_sizes = {}
        for feature in features:
            if feature not in self.df_item_idx.columns:
                print(f"Feature {feature} not found in df_item_idx = {self.df_item_idx.columns.tolist()}.")
            try:
                try:
                    self.vocab_sizes[feature] = len(self.mappings[f"{feature}"])
                except KeyError:
                    try:
                        self.vocab_sizes[feature] = len(self.mappings[f"{feature[:-1]}_entity2id"])
                    except KeyError:
                        self.vocab_sizes[feature] = len(self.mappings[f"{feature}_entity2id"])


            except KeyError:
                print(f"Feature {feature} not found in mappings = {self.mappings.keys()}.")

                self.vocab_sizes[feature] = -1
        return self.vocab_sizes

    def _df_index_preprocessing(self):
        i2x = self.mappings['item_to_idx']
        self.df_item_idx['item_id'] = self.df_item_idx['item_id'].map(i2x)
        # allow access on rows with index = item_id
        self.df_item_idx.set_index('item_id', inplace=True)

    # --- Batches Getters ---
    def build_user_unobserved_items(self):
        """
        Creates:
            self.user_observed_items: dict[user_id] -> set[item_id]
            self.user_unobserved_items: dict[user_id] -> np.ndarray[item_id]

        Unobserved items are:
            all catalog items \ user's observed training items
        """
        if self.train_coo is None:
            raise ValueError(
                "Train COO matrix not found. Please run train_test_split with format='coo' first."
            )

        user_rows = self.train_coo.row
        item_cols = self.train_coo.col
        ratings = self.train_coo.data

        all_items = np.arange(self.n_items)

        self.user_observed_items = {}

        for user, item, rating in zip(user_rows, item_cols, ratings):
            if rating == 0:
                continue

            if user not in self.user_observed_items:
                self.user_observed_items[user] = set()

            self.user_observed_items[user].add(item)

        self.user_unobserved_items = {}

        for user, observed_items in self.user_observed_items.items():
            observed_items_array = np.array(list(observed_items))

            self.user_unobserved_items[user] = np.setdiff1d(
                all_items,
                observed_items_array,
                assume_unique=False,
            )

    def get_train_batches_hybrid_observed_unobserved(
        self,
        batch_size: int,
        p_unobserved: float = 0.5,
        unobserved_score: float = 0.0,
        seed: int | None = None,
    ):
        """
        TODO: USE THIS ONE AS DEFAULT TRAIN BATCH GETTER!!!!!!!
        Creates one training pair per observed interaction.

        Positive item:
            observed item i with score r_ui > 0

        Negative / low item j:
            with probability p_unobserved:
                sample from user's unobserved items
            otherwise:
                sample from user's observed items with strictly lower score

        Returns:
            DataLoader with batches:
            [user_ids, pos_item_ids, neg_item_ids, pos_scores, neg_scores]
        """
        if self.train_coo is None:
            raise ValueError(
                "Train COO matrix not found. Please run train_test_split with format='coo' first."
            )

        if not hasattr(self, "user_unobserved_items"):
            self.build_user_unobserved_items()

        rng = np.random.default_rng(seed)

        user_rows = self.train_coo.row
        item_cols = self.train_coo.col
        ratings = self.train_coo.data

        user_to_observed_scored_items = {}

        for user, item, rating in zip(user_rows, item_cols, ratings):
            if rating == 0:
                raise ValueError("Train COO matrix contains zero ratings. Please ensure it only contains observed interactions with positive scores.")
                continue

            user_to_observed_scored_items.setdefault(user, []).append((item, rating))

        batch_users = []
        batch_pos_items = []
        batch_neg_items = []
        batch_pos_scores = []
        batch_neg_scores = []

        for user, item_ratings in user_to_observed_scored_items.items():

            unobserved_items = self.user_unobserved_items.get(user, np.array([]))

            for pos_item, pos_score in item_ratings:

                lower_observed_candidates = [
                    (item, score)
                    for item, score in item_ratings
                    if score < pos_score
                ]

                use_unobserved = rng.random() < p_unobserved

                if use_unobserved and len(unobserved_items) > 0:
                    neg_item = unobserved_items[
                        rng.integers(0, len(unobserved_items))
                    ]
                    neg_score = unobserved_score

                elif len(lower_observed_candidates) > 0:
                    neg_item, neg_score = lower_observed_candidates[
                        rng.integers(0, len(lower_observed_candidates))
                    ]

                elif len(unobserved_items) > 0:
                    # print(neg_score)
                    neg_item = unobserved_items[
                        rng.integers(0, len(unobserved_items))
                    ]
                    neg_score = unobserved_score

                else:
                    continue

                batch_users.append(user)
                batch_pos_items.append(pos_item)
                batch_neg_items.append(neg_item)
                batch_pos_scores.append(pos_score)
                batch_neg_scores.append(neg_score)

        if len(batch_users) == 0:
            raise ValueError("No valid training pairs could be sampled.")

        dataset = TensorDataset(
            torch.tensor(batch_users, dtype=torch.long),
            torch.tensor(batch_pos_items, dtype=torch.long),
            torch.tensor(batch_neg_items, dtype=torch.long),
            torch.tensor(batch_pos_scores, dtype=torch.float),
            torch.tensor(batch_neg_scores, dtype=torch.float),
        )

        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_train_batches_observed_pairs(self, batch_size: int, implicit_feedback: bool = True):
        """
        Build pairwise training batches using only observed items.

        For each user, creates pairs (item_i, item_j) such that:
            rating_i > rating_j

        Returns:
            DataLoader with batches:
            [user_ids, high_item_ids, low_item_ids, high_scores, low_scores]
        """
        if self.train_coo is None:
            raise ValueError(
                "Train COO matrix not found. Please run train_test_split with format='coo' first."
            )

        user_rows = self.train_coo.row
        item_cols = self.train_coo.col
        ratings = self.train_coo.data

        user_to_items = {}

        for user, item, rating in zip(user_rows, item_cols, ratings):
            if rating == 0:
                continue

            if user not in user_to_items:
                user_to_items[user] = []

            user_to_items[user].append((item, rating))

        batch_users = []
        batch_high_items = []
        batch_low_items = []
        batch_high_scores = []
        batch_low_scores = []

        for user, item_ratings in user_to_items.items():
            if len(item_ratings) < 2:
                continue

            for i in range(len(item_ratings)):
                item_i, rating_i = item_ratings[i]

                for j in range(len(item_ratings)):
                    item_j, rating_j = item_ratings[j]

                    if rating_i > rating_j:
                        batch_users.append(user)
                        batch_high_items.append(item_i)
                        batch_low_items.append(item_j)
                        batch_high_scores.append(rating_i)
                        batch_low_scores.append(rating_j)

        if len(batch_users) == 0:
            raise ValueError(
                "No valid observed preference pairs found. "
                "Each user needs at least two observed items with different scores."
            )

        dataset = TensorDataset(
            torch.tensor(batch_users, dtype=torch.long),
            torch.tensor(batch_high_items, dtype=torch.long),
            torch.tensor(batch_low_items, dtype=torch.long),
            torch.tensor(batch_high_scores, dtype=torch.float),
            torch.tensor(batch_low_scores, dtype=torch.float),
        )

        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def get_train_batches_observed_pairs_sampled(
        self,
        batch_size: int,
        seed: int | None = None,
    ):
        """
        Build pairwise training batches using only observed items.

        For each observed high item, sample ONE observed low item from the same user
        such that:

            high_score > low_score

        Returns:
            DataLoader with batches:
            [user_ids, high_item_ids, low_item_ids, high_scores, low_scores]
        """
        if self.train_coo is None:
            raise ValueError(
                "Train COO matrix not found. Please run train_test_split with format='coo' first."
            )

        rng = np.random.default_rng(seed)

        user_rows = self.train_coo.row
        item_cols = self.train_coo.col
        ratings = self.train_coo.data

        user_to_items = {}

        for user, item, rating in zip(user_rows, item_cols, ratings):
            if rating == 0:
                continue

            user_to_items.setdefault(user, []).append((item, rating))

        batch_users = []
        batch_high_items = []
        batch_low_items = []
        batch_high_scores = []
        batch_low_scores = []

        for user, item_ratings in user_to_items.items():
            if len(item_ratings) < 2:
                continue

            for high_item, high_score in item_ratings:
                candidate_lows = [
                    (low_item, low_score)
                    for low_item, low_score in item_ratings
                    if high_score > low_score
                ]

                if len(candidate_lows) == 0:
                    continue

                low_item, low_score = candidate_lows[
                    rng.integers(0, len(candidate_lows))
                ]

                batch_users.append(user)
                batch_high_items.append(high_item)
                batch_low_items.append(low_item)
                batch_high_scores.append(high_score)
                batch_low_scores.append(low_score)

        if len(batch_users) == 0:
            raise ValueError(
                "No valid observed preference pairs found. "
                "Each user needs at least two observed items with different scores."
            )

        dataset = TensorDataset(
            torch.tensor(batch_users, dtype=torch.long),
            torch.tensor(batch_high_items, dtype=torch.long),
            torch.tensor(batch_low_items, dtype=torch.long),
            torch.tensor(batch_high_scores, dtype=torch.float),
            torch.tensor(batch_low_scores, dtype=torch.float),
        )

        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def get_test_batches(self, implicit_feedback: bool = True):
        """
        Generate test batches from the test set.

        Returns:
            DataLoader: Test batches as a PyTorch DataLoader
        """
        users = []
        test_samples = []
        labels = []

        for user_id in self.dict_ui.keys():
            test_pos = self.dict_ui[user_id][1]
            user_ids = [user_id] * len(test_pos)
            test_items = np.array(list(test_pos))
            if implicit_feedback:
                user_labels = [1] * len(test_pos)
            else:
                user_labels = self.dict_ui[user_id][2]

            users.append(torch.tensor(user_ids, dtype=torch.long))
            test_samples.append(torch.tensor(test_items, dtype=torch.long))
            labels.append(torch.tensor(user_labels, dtype=torch.long))

        return list(zip(users, test_samples, labels))

    def clear_test_batches_cache(self):
        """
        Clear the cached test batches. Use this if you want to regenerate test batches.
        """
        cache_dir = os.path.join(self.path, 'test_batches_cache')
        cache_file = os.path.join(cache_dir, 'test_batches.pt')

        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"✅ Test batches cache cleared: {cache_file}")
        else:
            print(f"⚠️  No cached test batches found at {cache_file}")

    def get_test_batches_info(self):
        """
        Get information about cached test batches.

        Returns:
            dict: Information about the cache status
        """
        cache_dir = os.path.join(self.path, 'test_batches_cache')
        cache_file = os.path.join(cache_dir, 'test_batches.pt')

        info = {
            'cache_exists': os.path.exists(cache_file),
            'cache_path': cache_file
        }

        if info['cache_exists']:
            file_size = os.path.getsize(cache_file) / (1024 * 1024)  # Size in MB
            info['file_size_mb'] = f"{file_size:.2f} MB"

        return info

    def get_normalized_incidence_matrices(self) -> dict:
        incidence_matrices = {}
        for feature in self.train_features:
            feature_data = self.df_item_idx[feature]
            sparse_matrix = self._create_normalized_incidence_matrix(feature_data)
            # Convert scipy sparse matrix to PyTorch sparse tensor
            incidence_matrices[feature] = self._scipy_sparse_to_torch_sparse(sparse_matrix)
        return incidence_matrices

    def _scipy_sparse_to_torch_sparse(self, sparse_matrix) -> torch.sparse.FloatTensor:
        """Convert scipy sparse matrix to PyTorch sparse tensor"""
        sparse_matrix = sparse_matrix.tocoo()
        indices = torch.LongTensor(np.array([sparse_matrix.row, sparse_matrix.col]))
        values = torch.FloatTensor(sparse_matrix.data)
        shape = torch.Size(sparse_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def _create_normalized_incidence_matrix(self, feature_data) -> csr_matrix:
        # Create a binary incidence matrix for the given feature data
        num_items = self.df_item_idx.shape[0]
        num_features = self.vocab_sizes[feature_data.name]
        rows = []
        cols = []
        data = []

        for item_idx, features in feature_data.items():
            if features is None or (isinstance(features, float) and np.isnan(features)):
                continue
            for feature in features:
                rows.append(item_idx)
                cols.append(feature)
                data.append(1)

        # Create incidence matrix A
        A = csr_matrix((data, (rows, cols)), shape=(num_items, num_features))

        # Compute degree matrix D (sum of each row)
        degrees = np.array(A.sum(axis=1)).flatten()

        # Compute D^-1 (avoid division by zero)
        degrees_inv = np.zeros_like(degrees, dtype=np.float32)
        nonzero_mask = degrees > 0
        degrees_inv[nonzero_mask] = 1.0 / degrees[nonzero_mask]

        # Create D^-1 as a diagonal matrix and multiply: D^-1 * A
        D_inv = csr_matrix((degrees_inv, (np.arange(num_items), np.arange(num_items))))
        normalized_A = D_inv @ A

        return normalized_A


def create_high_low_test_pairs(batch_test):
    """
    Given a batch of test data (user_ids, item_ids, scores), create pairs of items for each user where one item has a higher score than the other.

    Args:
        batch_test: A tuple (user_ids, item_ids, scores) where each is a tensor of shape (batch_size,).
    Returns:
        high_ids: A list of item_ids that have higher scores.
        low_ids: A list of item_ids that have lower scores, corresponding to the high_ids.
    """
    # Create the test pairs such as score_item_1 > score_item_2

    high_ids = []
    low_ids = []

    item_ids = batch_test[1].numpy()
    scores = batch_test[2].numpy()

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i] > scores[j]:
                high_ids.append(item_ids[i])
                low_ids.append(item_ids[j])
    return high_ids, low_ids


def shuffled_batches(
    matrix,
    batch_size=256,
    seed=None,
    exclude_zeros=True,
    device="cpu",
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    coo = matrix.tocoo()

    indices = (
        np.flatnonzero(coo.data != 0)
        if exclude_zeros
        else np.arange(coo.nnz)
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]

        yield (
            torch.as_tensor(coo.row[idx], dtype=torch.long, device=device),
            torch.as_tensor(coo.col[idx], dtype=torch.long, device=device),
            torch.as_tensor(coo.data[idx], dtype=torch.float32, device=device),
        )