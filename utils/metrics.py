import numpy as np
import torch

@torch.inference_mode()
def pairwise_ranking_accuracy(model, one_user_test):
    user_ids, item_ids, ranks = one_user_test

    # Every unordered pair of items, without self-pairs.
    i, j = torch.triu_indices(
        len(item_ids), len(item_ids),
        offset=1,
        device=item_ids.device,
    )

    # Equal ground-truth ranks do not define a strict preference.
    comparable = ranks[i] != ranks[j]
    i, j = i[comparable], j[comparable]

    if i.numel() == 0:
        raise ValueError("No pairs with different ground-truth ranks.")

    # Put the preferred item first: smaller rank means more preferred.
    i_preferred = ranks[i] < ranks[j]
    pos_idx = torch.where(i_preferred, i, j)
    neg_idx = torch.where(i_preferred, j, i)

    was_training = model.training
    model.eval()

    try:
        pos_scores, neg_scores, _ = model(
            user_ids[pos_idx],
            item_ids[pos_idx],
            item_ids[neg_idx],
        )

        correct = pos_scores.reshape(-1) > neg_scores.reshape(-1)
        return 100.0 * correct.float().mean().item()
    finally:
        model.train(was_training)



import numpy as np
import torch


@torch.inference_mode()
def evaluate_topk(
    model,
    interactions,
    n_users,
    n_items,
    user_batch_size=32,
    device=None,
):
    """
    Score all items, excluding training items.

    Uses binary relevance and averages metrics over users with
    nonempty test sets. IDs must be contiguous, starting at zero.
    """
    if n_items < 1 or user_batch_size < 1:
        raise ValueError("n_items and user_batch_size must be positive.")

    train_by_user = {}
    test_by_user = {}

    for user_id, (train_ids, test_ids, _) in interactions.items():
        train_ids = set(train_ids)
        # Deduplicate so recall and ideal DCG use distinct relevant items.
        test_ids = np.unique(
            np.asarray(test_ids, dtype=np.int64).reshape(-1)
        )

        if train_ids.intersection(test_ids.tolist()):
            raise ValueError(f"Train/test overlap for user {user_id}.")

        train_by_user[user_id] = list(train_ids)
        if test_ids.size:
            test_by_user[user_id] = test_ids

    if not test_by_user:
        raise ValueError("No users have test items.")

    device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    all_items = torch.arange(n_items, device=device)

    ks = (10, 20)
    totals = {
        f"{metric}@{k}": 0.0
        for metric in ("HIT", "precision", "recall", "ndcg")
        for k in ks
    }

    # Rank 1 gets weight 1, rank 2 gets 1/log2(3), etc.
    discounts = 1.0 / np.log2(np.arange(2, max(ks) + 2))
    ideal_dcg_prefix = np.cumsum(discounts)

    was_training = model.training
    model.eval()

    try:
        for start in range(0, n_users, user_batch_size):
            stop = min(start + user_batch_size, n_users)
            users = torch.arange(start, stop, device=device)

            batch_users = users.repeat_interleave(n_items)
            batch_items = all_items.repeat(len(users))

            scores = model(batch_users, batch_items, None)
            scores = scores.reshape(len(users), n_items)

            # Exclude training items.
            for row, user_id in enumerate(range(start, stop)):
                train_ids = train_by_user.get(user_id, [])
                if train_ids:
                    ids = torch.tensor(
                        train_ids, dtype=torch.long, device=device
                    )
                    scores[row, ids] = -torch.inf

            top_scores, top_items = scores.topk(
                min(max(ks), n_items), dim=1
            )
            top_items = top_items.cpu().numpy()
            valid = torch.isfinite(top_scores).cpu().numpy()

            for row, user_id in enumerate(range(start, stop)):
                relevant = test_by_user.get(user_id)
                if relevant is None:
                    continue

                n_relevant = relevant.size
                hits = np.isin(top_items[row], relevant) & valid[row]

                for k in ks:
                    hits_k = hits[:k]
                    n_hits = int(hits_k.sum())

                    totals[f"HIT@{k}"] += float(n_hits > 0)
                    totals[f"precision@{k}"] += n_hits / k
                    totals[f"recall@{k}"] += n_hits / n_relevant

                    dcg = np.dot(
                        hits_k, discounts[:hits_k.size]
                    )
                    idcg = ideal_dcg_prefix[min(k, n_relevant) - 1]
                    totals[f"ndcg@{k}"] += float(dcg / idcg)

        return {
            name: total / len(test_by_user)
            for name, total in totals.items()
        }
    finally:
        model.train(was_training)



import math
import numpy as np
import torch


@torch.no_grad()
def evaluate_mse_rmse(
    model,
    interactions,
    n_users,
    n_items,
    user_batch_size=32,
    device=None,
    rating_scale=3.0,
):
    """
    interactions[user] = [set(train_ids), array(test_ids), array(test_ratings)]

    Returns pooled MSE and RMSE on the original rating scale.
    Evaluates all supplied test ratings, including zeros if present.
    """
    if n_users < 1 or n_items < 1 or user_batch_size < 1:
        raise ValueError(
            "n_users, n_items and user_batch_size must be positive."
        )

    test_by_user = {}

    for user_id, (train_ids, test_ids, test_ratings) in interactions.items():
        test_ids = np.asarray(test_ids, dtype=np.int64).reshape(-1)
        test_ratings = np.asarray(
            test_ratings, dtype=np.float64
        ).reshape(-1)

        if not 0 <= user_id < n_users:
            raise ValueError(f"Invalid user ID: {user_id}.")

        if test_ids.size != test_ratings.size:
            raise ValueError(
                f"Item/rating length mismatch for user {user_id}."
            )

        if np.any((test_ids < 0) | (test_ids >= n_items)):
            raise ValueError(f"Invalid test item ID for user {user_id}.")

        if set(train_ids).intersection(test_ids.tolist()):
            raise ValueError(f"Train/test overlap for user {user_id}.")

        if not np.isfinite(test_ratings).all():
            raise ValueError(f"Nonfinite test rating for user {user_id}.")

        if test_ids.size:
            test_by_user[user_id] = (test_ids, test_ratings)

    if not test_by_user:
        raise ValueError("No users have test items.")

    device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )

    user_ids = sorted(test_by_user)
    total_squared_error = 0.0
    n_test = 0

    was_training = model.training
    model.eval()

    try:
        for start in range(0, len(user_ids), user_batch_size):
            batch_ids = user_ids[start:start + user_batch_size]

            # Build aligned user-item-rating arrays for test pairs only.
            users = np.concatenate([
                np.full(
                    test_by_user[u][0].size, u, dtype=np.int64
                )
                for u in batch_ids
            ])
            items = np.concatenate([
                test_by_user[u][0] for u in batch_ids
            ])
            ratings = np.concatenate([
                test_by_user[u][1] for u in batch_ids
            ])

            users = torch.as_tensor(
                users, dtype=torch.long, device=device
            )
            items = torch.as_tensor(
                items, dtype=torch.long, device=device
            )
            ratings = torch.as_tensor(
                ratings, dtype=torch.float64, device=device
            )

            scores = model(users, items, None).reshape(-1).to(torch.float64)

            if scores.numel() != ratings.numel():
                raise ValueError("Expected one model score per test pair.")

            if not torch.isfinite(scores).all():
                raise ValueError("Model produced nonfinite scores.")

            predicted_ratings = rating_scale * scores
            squared_errors = (predicted_ratings - ratings).square()
            absolute_errors = (predicted_ratings - ratings).abs()

            total_squared_error += squared_errors.sum().item()
            total_absolute_error += absolute_errors.sum().item()

            n_test += ratings.numel()

        mse = total_squared_error / n_test
        mae = total_absolute_error / n_test

        return {
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "MAE": mae,
        }

    finally:
        model.train(was_training)