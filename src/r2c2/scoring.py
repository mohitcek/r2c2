"""Post-hoc consistency scoring with UQLM's noncontradiction scorer.

Caching means the model re-reads the prompt cheaply but still re-samples the
answer; this checks whether the answers agree. Scoring runs locally on answers you
already have — zero extra API calls — but it does pull down
microsoft/deberta-large-mnli (~1.6 GB) the first time, and it's slow on CPU. The
uqlm import is deferred so the rest of the package never pays for it.

Scores are rankings, not calibrated probabilities, and consistency is not
correctness: a model that hedges identically N times scores 1.00.
"""

from __future__ import annotations

import asyncio
import inspect
import statistics
from collections.abc import Sequence


def consistency_scores_batch(groups: Sequence[Sequence[str]]) -> list[list[float]]:
    """Leave-one-out noncontradiction scores for several answer sets in one NLI pass.

    Each answer is scored against the others in its group rather than anchoring on
    the first sample, so the result doesn't depend on which answer happened to come
    first. Batching everything into a single scorer call matters: the NLI model is
    the slow part, not the bookkeeping.
    """
    from uqlm import BlackBoxUQ  # deferred: pulls transformers/torch

    responses, sampled, spans = [], [], []
    for answers in groups:
        answers = list(answers)
        if len(answers) < 2:
            raise ValueError("need at least 2 answers per group to score consistency")
        start = len(responses)
        for i, answer in enumerate(answers):
            responses.append(answer)
            sampled.append(answers[:i] + answers[i + 1 :])
        spans.append((start, len(responses)))

    # No llm= : post-hoc scoring of pre-generated responses doesn't need one.
    # use_best=False keeps the original answers instead of swapping in the most
    # consistent one — we want the score, not a fix.
    scorer = BlackBoxUQ(scorers=["noncontradiction"], use_best=False)
    result = scorer.score(responses=responses, sampled_responses=sampled)
    if inspect.isawaitable(result):  # sync in uqlm 0.6.4, coroutine in other versions
        result = asyncio.run(result)
    scores = result.to_df()["noncontradiction"].tolist()
    return [[float(s) for s in scores[a:b]] for a, b in spans]


def consistency_scores(answers: Sequence[str]) -> list[float]:
    """Leave-one-out noncontradiction score for each answer in a single set."""
    return consistency_scores_batch([answers])[0]


def confidence(answers: Sequence[str]) -> float:
    """Mean leave-one-out noncontradiction score — the one-number confidence signal."""
    return statistics.fmean(consistency_scores(answers))
