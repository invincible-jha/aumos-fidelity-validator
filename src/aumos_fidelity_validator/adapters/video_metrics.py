"""Video metrics adapter — evaluates video synthesis fidelity.

Implements VideoMetricsProtocol measuring per-frame LPIPS, optical flow
consistency, temporal coherence, scene transition accuracy, and overall
video fidelity aggregation. All heavy computation dispatched to thread pool.
"""

import asyncio
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Type alias: video as (num_frames, H, W, C) uint8 numpy array
VideoArray = np.ndarray


class VideoMetricsEvaluator:
    """Evaluates video synthesis quality across temporal and spatial dimensions.

    Computes per-frame LPIPS, optical flow consistency between real and
    synthetic sequences, frame-to-frame temporal coherence, scene transition
    detection accuracy, and an aggregated video fidelity score.
    """

    # Maximum number of frames to sample for efficiency
    _MAX_FRAME_SAMPLE: int = 64
    # Minimum frames for optical flow computation
    _MIN_FLOW_FRAMES: int = 4

    async def evaluate(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Run full video fidelity evaluation.

        Args:
            real_video: Real video frames, shape (T, H, W, C), uint8, RGB.
            synthetic_video: Synthetic video frames, same format.

        Returns:
            Report dict with lpips_score, optical_flow_score, temporal_score,
            scene_transition_score, and overall_score.
        """
        logger.info(
            "Running video metrics evaluation",
            real_frames=real_video.shape[0] if len(real_video.shape) > 0 else 0,
            synthetic_frames=synthetic_video.shape[0] if len(synthetic_video.shape) > 0 else 0,
        )
        loop = asyncio.get_running_loop()

        lpips_result = await loop.run_in_executor(
            None, self._compute_per_frame_lpips, real_video, synthetic_video
        )
        flow_result = await loop.run_in_executor(
            None, self._compute_optical_flow_consistency, real_video, synthetic_video
        )
        temporal_result = await loop.run_in_executor(
            None, self._compute_temporal_coherence, real_video, synthetic_video
        )
        transition_result = await loop.run_in_executor(
            None, self._compute_scene_transition_accuracy, real_video, synthetic_video
        )

        lpips_score = float(1.0 - min(lpips_result.get("mean_lpips", 1.0), 1.0))
        flow_score = flow_result.get("flow_consistency_score", 0.0)
        temporal_score = temporal_result.get("temporal_coherence_score", 0.0)
        transition_score = transition_result.get("transition_detection_score", 0.0)

        overall_score = (
            lpips_score * 0.35
            + flow_score * 0.30
            + temporal_score * 0.25
            + transition_score * 0.10
        )

        return {
            "overall_score": float(overall_score),
            "per_frame_lpips": lpips_result,
            "lpips_score": float(lpips_score),
            "optical_flow_consistency": flow_result,
            "flow_score": float(flow_score),
            "temporal_coherence": temporal_result,
            "temporal_score": float(temporal_score),
            "scene_transitions": transition_result,
            "transition_score": float(transition_score),
        }

    def _compute_per_frame_lpips(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Compute per-frame perceptual similarity (LPIPS proxy via SSIM).

        Args:
            real_video: Real video frames (T, H, W, C).
            synthetic_video: Synthetic video frames (T, H, W, C).

        Returns:
            Dict with mean_lpips, std_lpips, and per-frame LPIPS values.
        """
        try:
            from skimage.metrics import structural_similarity as ssim  # type: ignore[import]

            num_frames = min(
                real_video.shape[0],
                synthetic_video.shape[0],
                self._MAX_FRAME_SAMPLE,
            )
            # Sample frames uniformly
            frame_indices = np.linspace(0, num_frames - 1, num_frames, dtype=int)
            per_frame_scores: list[float] = []

            for idx in frame_indices:
                real_frame = real_video[idx]
                synth_frame = synthetic_video[idx]

                if real_frame.shape != synth_frame.shape:
                    from skimage.transform import resize  # type: ignore[import]
                    synth_frame = (resize(synth_frame.astype(float), real_frame.shape) * 255).astype(np.uint8)

                channel_axis = 2 if len(real_frame.shape) == 3 else None
                ssim_score = float(
                    ssim(real_frame, synth_frame, channel_axis=channel_axis, data_range=255)
                )
                # Convert SSIM to LPIPS proxy: LPIPS ~ 1 - SSIM
                per_frame_scores.append(float(1.0 - ssim_score))

            return {
                "mean_lpips": float(np.mean(per_frame_scores)) if per_frame_scores else 0.5,
                "std_lpips": float(np.std(per_frame_scores)) if per_frame_scores else 0.0,
                "frames_evaluated": len(per_frame_scores),
                "method": "ssim_proxy",
            }

        except ImportError:
            logger.warning("scikit-image not available — using pixel MSE proxy for LPIPS")
            return self._compute_per_frame_mse_proxy(real_video, synthetic_video)
        except Exception as exc:
            logger.warning("Per-frame LPIPS computation failed", error=str(exc))
            return {"mean_lpips": 0.5, "error": str(exc)}

    def _compute_per_frame_mse_proxy(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Fallback per-frame quality via normalised pixel MSE.

        Args:
            real_video: Real frames.
            synthetic_video: Synthetic frames.

        Returns:
            Dict with mean_lpips.
        """
        num_frames = min(real_video.shape[0], synthetic_video.shape[0], self._MAX_FRAME_SAMPLE)
        per_frame_scores: list[float] = []

        for i in range(num_frames):
            real_frame = real_video[i].astype(float) / 255.0
            synth_frame = synthetic_video[i].astype(float) / 255.0
            min_shape = tuple(min(a, b) for a, b in zip(real_frame.shape, synth_frame.shape))
            r = real_frame[: min_shape[0], : min_shape[1]]
            s = synth_frame[: min_shape[0], : min_shape[1]]
            mse = float(np.mean((r - s) ** 2))
            per_frame_scores.append(mse)

        return {
            "mean_lpips": float(np.mean(per_frame_scores)) if per_frame_scores else 0.5,
            "method": "pixel_mse_proxy",
        }

    def _compute_optical_flow_consistency(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Measure optical flow consistency between real and synthetic video.

        Computes dense optical flow between consecutive frames using Lucas-Kanade
        and measures how similarly real and synthetic videos move over time.

        Args:
            real_video: Real video frames (T, H, W, C).
            synthetic_video: Synthetic video frames (T, H, W, C).

        Returns:
            Dict with flow_consistency_score and mean_flow_magnitude_difference.
        """
        try:
            import cv2  # type: ignore[import]

            num_frames = min(
                real_video.shape[0],
                synthetic_video.shape[0],
                self._MAX_FRAME_SAMPLE,
            )

            if num_frames < self._MIN_FLOW_FRAMES:
                return {"flow_consistency_score": 0.5, "note": "insufficient_frames"}

            real_magnitudes: list[float] = []
            synth_magnitudes: list[float] = []

            for i in range(num_frames - 1):
                real_gray1 = cv2.cvtColor(real_video[i], cv2.COLOR_RGB2GRAY)
                real_gray2 = cv2.cvtColor(real_video[i + 1], cv2.COLOR_RGB2GRAY)
                synth_gray1 = cv2.cvtColor(synthetic_video[i], cv2.COLOR_RGB2GRAY)
                synth_gray2 = cv2.cvtColor(synthetic_video[i + 1], cv2.COLOR_RGB2GRAY)

                real_flow = cv2.calcOpticalFlowFarneback(
                    real_gray1, real_gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                synth_flow = cv2.calcOpticalFlowFarneback(
                    synth_gray1, synth_gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                real_mag = float(np.mean(np.sqrt(real_flow[..., 0] ** 2 + real_flow[..., 1] ** 2)))
                synth_mag = float(np.mean(np.sqrt(synth_flow[..., 0] ** 2 + synth_flow[..., 1] ** 2)))
                real_magnitudes.append(real_mag)
                synth_magnitudes.append(synth_mag)

            real_arr = np.array(real_magnitudes)
            synth_arr = np.array(synth_magnitudes)
            diff = np.abs(real_arr - synth_arr)
            scale = real_arr.mean() + 1e-8
            normalised_diff = float(np.mean(diff / scale))
            consistency_score = float(max(0.0, 1.0 - normalised_diff))

            return {
                "flow_consistency_score": consistency_score,
                "mean_real_flow_magnitude": float(np.mean(real_magnitudes)),
                "mean_synthetic_flow_magnitude": float(np.mean(synth_magnitudes)),
                "mean_flow_magnitude_difference": float(np.mean(diff)),
            }

        except ImportError:
            logger.warning("OpenCV not available — using frame difference proxy for optical flow")
            return self._compute_flow_proxy(real_video, synthetic_video)
        except Exception as exc:
            logger.warning("Optical flow computation failed", error=str(exc))
            return {"flow_consistency_score": 0.0, "error": str(exc)}

    def _compute_flow_proxy(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Fallback flow consistency via frame difference magnitudes.

        Args:
            real_video: Real frames.
            synthetic_video: Synthetic frames.

        Returns:
            Dict with flow_consistency_score.
        """
        num_frames = min(real_video.shape[0], synthetic_video.shape[0], self._MAX_FRAME_SAMPLE)
        real_diffs: list[float] = []
        synth_diffs: list[float] = []

        for i in range(num_frames - 1):
            real_diff = float(np.mean(np.abs(real_video[i + 1].astype(float) - real_video[i].astype(float))))
            synth_diff = float(np.mean(np.abs(synthetic_video[i + 1].astype(float) - synthetic_video[i].astype(float))))
            real_diffs.append(real_diff)
            synth_diffs.append(synth_diff)

        if not real_diffs:
            return {"flow_consistency_score": 0.5}

        ratio = np.array(synth_diffs) / (np.array(real_diffs) + 1e-8)
        score = float(1.0 - np.mean(np.clip(np.abs(ratio - 1.0), 0, 1)))
        return {"flow_consistency_score": score, "method": "frame_diff_proxy"}

    def _compute_temporal_coherence(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Measure frame-to-frame temporal stability within each video.

        A coherent video has smooth transitions between frames. We measure
        this as the average cosine similarity between consecutive frame embeddings.

        Args:
            real_video: Real video frames (T, H, W, C).
            synthetic_video: Synthetic video frames (T, H, W, C).

        Returns:
            Dict with temporal_coherence_score for real and synthetic.
        """
        try:
            def _coherence(video: VideoArray) -> float:
                num_frames = min(video.shape[0], self._MAX_FRAME_SAMPLE)
                if num_frames < 2:
                    return 1.0
                similarities: list[float] = []
                for i in range(num_frames - 1):
                    flat_a = video[i].astype(float).ravel()
                    flat_b = video[i + 1].astype(float).ravel()
                    cos_sim = float(
                        np.dot(flat_a, flat_b)
                        / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-8)
                    )
                    similarities.append(max(0.0, cos_sim))
                return float(np.mean(similarities))

            real_coherence = _coherence(real_video)
            synth_coherence = _coherence(synthetic_video)
            # Score measures how well synthetic coherence matches real coherence
            coherence_score = float(1.0 - min(abs(real_coherence - synth_coherence), 1.0))

            return {
                "temporal_coherence_score": coherence_score,
                "real_coherence": real_coherence,
                "synthetic_coherence": synth_coherence,
            }

        except Exception as exc:
            logger.warning("Temporal coherence computation failed", error=str(exc))
            return {"temporal_coherence_score": 0.0, "error": str(exc)}

    def _compute_scene_transition_accuracy(
        self,
        real_video: VideoArray,
        synthetic_video: VideoArray,
    ) -> dict[str, Any]:
        """Detect scene transitions and compare accuracy between real and synthetic.

        A scene transition is a sudden large change between consecutive frames.
        We detect transitions in both videos and measure how many real
        transitions are also present in the synthetic video.

        Args:
            real_video: Real video frames (T, H, W, C).
            synthetic_video: Synthetic video frames (T, H, W, C).

        Returns:
            Dict with transition_detection_score and transition counts.
        """
        try:
            def _detect_transitions(video: VideoArray, threshold: float = 20.0) -> list[int]:
                transitions: list[int] = []
                num_frames = min(video.shape[0], self._MAX_FRAME_SAMPLE)
                for i in range(num_frames - 1):
                    mean_diff = float(
                        np.mean(np.abs(video[i + 1].astype(float) - video[i].astype(float)))
                    )
                    if mean_diff > threshold:
                        transitions.append(i)
                return transitions

            real_transitions = set(_detect_transitions(real_video))
            synth_transitions = set(_detect_transitions(synthetic_video))

            if not real_transitions and not synth_transitions:
                return {
                    "transition_detection_score": 1.0,
                    "real_transition_count": 0,
                    "synthetic_transition_count": 0,
                    "matched_transitions": 0,
                }

            # Allow 1-frame tolerance for matching
            matched = 0
            for rt in real_transitions:
                if any(abs(rt - st) <= 1 for st in synth_transitions):
                    matched += 1

            precision = matched / max(len(synth_transitions), 1)
            recall = matched / max(len(real_transitions), 1)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            return {
                "transition_detection_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "real_transition_count": len(real_transitions),
                "synthetic_transition_count": len(synth_transitions),
                "matched_transitions": matched,
            }

        except Exception as exc:
            logger.warning("Scene transition computation failed", error=str(exc))
            return {"transition_detection_score": 0.0, "error": str(exc)}
