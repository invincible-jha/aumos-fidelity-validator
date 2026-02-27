"""Text metrics adapter — evaluates text generation quality.

Implements TextMetricsProtocol using sacrebleu (BLEU), rouge-score (ROUGE),
sentence-transformers (semantic similarity), and cross-entropy perplexity
estimation. All heavy computation is dispatched to a thread pool.
"""

import asyncio
import math
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class TextMetricsEvaluator:
    """Evaluates quality of generated text against reference corpora.

    Computes BLEU (1-4 gram), ROUGE-1/2/L, semantic similarity via
    sentence-transformer embeddings, text coherence via sliding window
    cosine similarity, and perplexity estimation via cross-entropy.
    """

    # Sliding window size for coherence scoring
    _COHERENCE_WINDOW: int = 3
    # Minimum texts required to produce meaningful scores
    _MIN_TEXT_COUNT: int = 5

    async def evaluate(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> dict[str, Any]:
        """Run all text quality metrics.

        Args:
            real_texts: Reference (ground-truth) text samples.
            synthetic_texts: Generated text samples to evaluate.
            model_name: Sentence-transformer model for semantic similarity.

        Returns:
            Report dict with bleu, rouge, semantic_similarity, coherence,
            perplexity_ratio, and overall_score.
        """
        logger.info(
            "Running text metrics evaluation",
            real_count=len(real_texts),
            synthetic_count=len(synthetic_texts),
            model=model_name,
        )

        if len(real_texts) < self._MIN_TEXT_COUNT or len(synthetic_texts) < self._MIN_TEXT_COUNT:
            logger.warning(
                "Insufficient text samples for evaluation",
                real_count=len(real_texts),
                synthetic_count=len(synthetic_texts),
                minimum=self._MIN_TEXT_COUNT,
            )

        loop = asyncio.get_running_loop()

        bleu_result, rouge_result = await asyncio.gather(
            loop.run_in_executor(None, self._compute_bleu, real_texts, synthetic_texts),
            loop.run_in_executor(None, self._compute_rouge, real_texts, synthetic_texts),
        )

        semantic_result = await loop.run_in_executor(
            None, self._compute_semantic_similarity, real_texts, synthetic_texts, model_name
        )

        coherence_real = await loop.run_in_executor(
            None, self._compute_coherence, real_texts, model_name
        )
        coherence_synth = await loop.run_in_executor(
            None, self._compute_coherence, synthetic_texts, model_name
        )
        coherence_ratio = float(coherence_synth / max(coherence_real, 1e-6))
        coherence_score = float(min(coherence_ratio, 1.0))

        perplexity_score = await loop.run_in_executor(
            None, self._compute_perplexity_similarity, real_texts, synthetic_texts
        )

        overall_score = (
            bleu_result.get("bleu_avg", 0.0) * 0.20
            + rouge_result.get("rouge_l_fmeasure", 0.0) * 0.25
            + semantic_result.get("mean_cosine_similarity", 0.0) * 0.35
            + coherence_score * 0.10
            + perplexity_score * 0.10
        )

        return {
            "overall_score": float(overall_score),
            "bleu": bleu_result,
            "rouge": rouge_result,
            "semantic_similarity": semantic_result,
            "coherence": {
                "real_coherence": float(coherence_real),
                "synthetic_coherence": float(coherence_synth),
                "coherence_score": float(coherence_score),
            },
            "perplexity": {
                "perplexity_similarity_score": float(perplexity_score),
            },
        }

    def _compute_bleu(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
    ) -> dict[str, Any]:
        """Compute corpus-level BLEU-1 through BLEU-4 scores.

        Args:
            real_texts: Reference sentences.
            synthetic_texts: Hypothesis sentences.

        Returns:
            Dict with bleu_1 through bleu_4 and bleu_avg scores.
        """
        try:
            import sacrebleu  # type: ignore[import]

            # sacrebleu expects list-of-reference-lists for corpus_bleu
            references = [real_texts]
            hypotheses = synthetic_texts[: len(real_texts)]

            bleu1 = sacrebleu.corpus_bleu(hypotheses, references, max_ngram_order=1).score / 100.0
            bleu2 = sacrebleu.corpus_bleu(hypotheses, references, max_ngram_order=2).score / 100.0
            bleu3 = sacrebleu.corpus_bleu(hypotheses, references, max_ngram_order=3).score / 100.0
            bleu4 = sacrebleu.corpus_bleu(hypotheses, references, max_ngram_order=4).score / 100.0

            return {
                "bleu_1": float(bleu1),
                "bleu_2": float(bleu2),
                "bleu_3": float(bleu3),
                "bleu_4": float(bleu4),
                "bleu_avg": float((bleu1 + bleu2 + bleu3 + bleu4) / 4.0),
            }
        except ImportError:
            logger.warning("sacrebleu not installed — falling back to NLTK BLEU")
            return self._compute_bleu_nltk(real_texts, synthetic_texts)
        except Exception as exc:
            logger.warning("BLEU computation failed", error=str(exc))
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bleu_avg": 0.0}

    def _compute_bleu_nltk(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
    ) -> dict[str, Any]:
        """Fallback BLEU computation via NLTK.

        Args:
            real_texts: Reference sentences.
            synthetic_texts: Hypothesis sentences.

        Returns:
            BLEU score dict.
        """
        try:
            from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu  # type: ignore[import]

            smoothie = SmoothingFunction().method4
            refs = [[ref.split()] for ref in real_texts]
            hyps = [h.split() for h in synthetic_texts[: len(real_texts)]]

            scores = []
            for ngram in range(1, 5):
                weights = tuple(1.0 / ngram if i < ngram else 0.0 for i in range(1, 5))
                score = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoothie)
                scores.append(float(score))

            return {
                "bleu_1": scores[0],
                "bleu_2": scores[1],
                "bleu_3": scores[2],
                "bleu_4": scores[3],
                "bleu_avg": float(sum(scores) / 4.0),
            }
        except Exception as exc:
            logger.warning("NLTK BLEU fallback failed", error=str(exc))
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bleu_avg": 0.0}

    def _compute_rouge(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
    ) -> dict[str, Any]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

        Args:
            real_texts: Reference sentences.
            synthetic_texts: Hypothesis sentences.

        Returns:
            Dict with rouge_1, rouge_2, rouge_l precision/recall/fmeasure.
        """
        try:
            from rouge_score import rouge_scorer  # type: ignore[import]

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

            r1_p, r1_r, r1_f = [], [], []
            r2_p, r2_r, r2_f = [], [], []
            rl_p, rl_r, rl_f = [], [], []

            pairs = list(zip(real_texts, synthetic_texts))
            for reference, hypothesis in pairs:
                scores_dict = scorer.score(reference, hypothesis)
                r1_p.append(scores_dict["rouge1"].precision)
                r1_r.append(scores_dict["rouge1"].recall)
                r1_f.append(scores_dict["rouge1"].fmeasure)
                r2_p.append(scores_dict["rouge2"].precision)
                r2_r.append(scores_dict["rouge2"].recall)
                r2_f.append(scores_dict["rouge2"].fmeasure)
                rl_p.append(scores_dict["rougeL"].precision)
                rl_r.append(scores_dict["rougeL"].recall)
                rl_f.append(scores_dict["rougeL"].fmeasure)

            def _avg(vals: list[float]) -> float:
                return float(sum(vals) / len(vals)) if vals else 0.0

            return {
                "rouge_1_precision": _avg(r1_p),
                "rouge_1_recall": _avg(r1_r),
                "rouge_1_fmeasure": _avg(r1_f),
                "rouge_2_precision": _avg(r2_p),
                "rouge_2_recall": _avg(r2_r),
                "rouge_2_fmeasure": _avg(r2_f),
                "rouge_l_precision": _avg(rl_p),
                "rouge_l_recall": _avg(rl_r),
                "rouge_l_fmeasure": _avg(rl_f),
            }
        except ImportError:
            logger.warning("rouge_score not installed — ROUGE scores unavailable")
            return {k: 0.0 for k in [
                "rouge_1_precision", "rouge_1_recall", "rouge_1_fmeasure",
                "rouge_2_precision", "rouge_2_recall", "rouge_2_fmeasure",
                "rouge_l_precision", "rouge_l_recall", "rouge_l_fmeasure",
            ]}
        except Exception as exc:
            logger.warning("ROUGE computation failed", error=str(exc))
            return {k: 0.0 for k in [
                "rouge_1_fmeasure", "rouge_2_fmeasure", "rouge_l_fmeasure",
            ]}

    def _compute_semantic_similarity(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
        model_name: str,
    ) -> dict[str, Any]:
        """Compute mean cosine similarity between sentence embeddings.

        Encodes all texts with a sentence-transformer and computes pairwise
        cosine similarity between paired real and synthetic sentences.

        Args:
            real_texts: Reference sentences.
            synthetic_texts: Generated sentences.
            model_name: Sentence-transformer model identifier.

        Returns:
            Dict with mean_cosine_similarity and std_cosine_similarity.
        """
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            model = SentenceTransformer(model_name)
            sample_size = min(len(real_texts), len(synthetic_texts), 500)
            real_sample = real_texts[:sample_size]
            synth_sample = synthetic_texts[:sample_size]

            real_embeddings = model.encode(real_sample, convert_to_numpy=True, show_progress_bar=False)
            synth_embeddings = model.encode(synth_sample, convert_to_numpy=True, show_progress_bar=False)

            # Paired cosine similarities
            real_norms = np.linalg.norm(real_embeddings, axis=1, keepdims=True)
            synth_norms = np.linalg.norm(synth_embeddings, axis=1, keepdims=True)
            real_normed = real_embeddings / (real_norms + 1e-8)
            synth_normed = synth_embeddings / (synth_norms + 1e-8)
            cosine_sims = np.sum(real_normed * synth_normed, axis=1)

            return {
                "mean_cosine_similarity": float(np.mean(cosine_sims)),
                "std_cosine_similarity": float(np.std(cosine_sims)),
                "min_cosine_similarity": float(np.min(cosine_sims)),
                "max_cosine_similarity": float(np.max(cosine_sims)),
                "sample_size": sample_size,
            }
        except ImportError:
            logger.warning("sentence-transformers not installed — semantic similarity unavailable")
            return {"mean_cosine_similarity": 0.0, "std_cosine_similarity": 0.0}
        except Exception as exc:
            logger.warning("Semantic similarity computation failed", error=str(exc))
            return {"mean_cosine_similarity": 0.0, "std_cosine_similarity": 0.0}

    def _compute_coherence(
        self,
        texts: list[str],
        model_name: str,
    ) -> float:
        """Score text coherence via sliding window embedding similarity.

        Encodes consecutive text windows and measures embedding continuity.
        High coherence means adjacent texts have similar embeddings.

        Args:
            texts: List of texts to evaluate coherence over.
            model_name: Sentence-transformer model identifier.

        Returns:
            Mean coherence score in [0, 1].
        """
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            if len(texts) < self._COHERENCE_WINDOW + 1:
                return 1.0

            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts[:200], convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normed = embeddings / (norms + 1e-8)

            similarities: list[float] = []
            window = self._COHERENCE_WINDOW
            for i in range(len(normed) - window):
                window_mean = normed[i : i + window].mean(axis=0)
                next_vec = normed[i + window]
                sim = float(np.dot(window_mean, next_vec))
                similarities.append(max(0.0, sim))

            return float(sum(similarities) / len(similarities)) if similarities else 1.0
        except Exception:  # noqa: BLE001
            return 0.5

    def _compute_perplexity_similarity(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
    ) -> float:
        """Estimate perplexity similarity via unigram cross-entropy.

        Builds a unigram language model from real texts and evaluates
        the cross-entropy of synthetic texts under that model. Returns
        a normalised score where 1.0 means equal perplexity.

        Args:
            real_texts: Reference texts to build LM from.
            synthetic_texts: Texts to evaluate.

        Returns:
            Perplexity similarity score in [0, 1].
        """
        try:
            from collections import Counter

            def _build_unigram_lm(texts: list[str]) -> dict[str, float]:
                counts: Counter[str] = Counter()
                total = 0
                for text in texts:
                    tokens = text.lower().split()
                    counts.update(tokens)
                    total += len(tokens)
                vocab_size = len(counts)
                # Add-1 (Laplace) smoothing
                return {
                    token: (count + 1) / (total + vocab_size)
                    for token, count in counts.items()
                }

            def _cross_entropy(texts: list[str], lm: dict[str, float]) -> float:
                unk_prob = 1e-8
                total_log_prob = 0.0
                total_tokens = 0
                for text in texts:
                    tokens = text.lower().split()
                    for token in tokens:
                        prob = lm.get(token, unk_prob)
                        total_log_prob += math.log2(prob)
                    total_tokens += len(tokens)
                return -total_log_prob / max(total_tokens, 1)

            real_lm = _build_unigram_lm(real_texts)
            real_entropy = _cross_entropy(real_texts, real_lm)
            synth_entropy = _cross_entropy(synthetic_texts, real_lm)

            if real_entropy <= 0:
                return 1.0
            ratio = synth_entropy / real_entropy
            # Score is highest when ratio = 1.0 (same perplexity)
            return float(1.0 - min(abs(ratio - 1.0), 1.0))
        except Exception as exc:
            logger.warning("Perplexity estimation failed", error=str(exc))
            return 0.5
