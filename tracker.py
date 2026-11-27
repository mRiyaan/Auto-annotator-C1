from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    Linear bounding-box Kalman Filter in state space [cx, cy, w, h, vx, vy, vw, vh].

    Using direct pixel width/height instead of nonlinear area/aspect-ratio (s, r) avoids
    division instability and prevents aspect-ratio velocity runaway during coasting frames.
    """

    count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        score: float = 1.0,
        label: str = "target",
        event_id: int = 0,
    ):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        # Trust direct VLM / smoothly interpolated measurements cleanly over constant-velocity lag.
        self.kf.processNoiseCov = np.diag(
            [0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.08, 0.08]
        ).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([0.05, 0.05, 0.08, 0.08]).astype(np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.5

        cx, cy, w, h = self._bbox_to_xywh(bbox)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ).reshape(-1, 1)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.score = float(score)
        self.label = label
        self.event_id = int(event_id)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = np.array(bbox, dtype=np.float32).copy()
        self.last_observed_bbox = np.array(bbox, dtype=np.float32).copy()
        self.is_lost = False

    @staticmethod
    def _bbox_to_xywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        w = max(12.0, float(bbox[2] - bbox[0]))
        h = max(12.0, float(bbox[3] - bbox[1]))
        cx = float(bbox[0]) + w / 2.0
        cy = float(bbox[1]) + h / 2.0
        return cx, cy, w, h

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> np.ndarray:
        cx = float(state[0, 0])
        cy = float(state[1, 0])
        w = max(12.0, float(state[2, 0]))
        h = max(12.0, float(state[3, 0]))
        return np.array(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32
        )

    def predict(self) -> np.ndarray:
        coasting = self.time_since_update > 0
        if coasting:
            # Dampen both translational and size velocities strictly while ungrounded.
            self.kf.statePost[4, 0] *= 0.78 if self.time_since_update < 8 else 0.0
            self.kf.statePost[5, 0] *= 0.78 if self.time_since_update < 8 else 0.0
            self.kf.statePost[6, 0] *= 0.25 if self.time_since_update < 3 else 0.0
            self.kf.statePost[7, 0] *= 0.25 if self.time_since_update < 3 else 0.0
        else:
            self.kf.statePost[6, 0] *= 0.70
            self.kf.statePost[7, 0] *= 0.70

        p = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        if coasting:
            # Bound box dimensions near last observed size so lost boxes never deform horizontally.
            _, _, obs_w, obs_h = self._bbox_to_xywh(self.last_observed_bbox)
            p[2, 0] = float(np.clip(p[2, 0], max(12.0, 0.80 * obs_w), 1.25 * obs_w))
            p[3, 0] = float(np.clip(p[3, 0], max(12.0, 0.80 * obs_h), 1.25 * obs_h))
        else:
            p[2, 0] = max(12.0, float(p[2, 0]))
            p[3, 0] = max(12.0, float(p[3, 0]))

        self.kf.statePost[0:4, 0] = p[0:4, 0]
        self.last_bbox = self._state_to_bbox(p)
        return self.last_bbox

    def update(
        self,
        bbox: np.ndarray,
        score: float = 1.0,
        label: Optional[str] = None,
        event_id: Optional[int] = None,
    ) -> None:
        was_coasting = self.time_since_update > 2
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        self.is_lost = False
        if label:
            self.label = label
        if event_id is not None and event_id > 0:
            self.event_id = int(event_id)

        cx, cy, w, h = self._bbox_to_xywh(bbox)
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(-1, 1)

        if was_coasting:
            # Re-anchor cleanly on recovery from lost frames without velocity overshoot.
            self.kf.statePost[0:4, 0] = measurement[:, 0]
            self.kf.statePost[4:8, 0] *= 0.15
        else:
            self.kf.correct(measurement)
            self.kf.statePost[2, 0] = max(12.0, float(self.kf.statePost[2, 0]))
            self.kf.statePost[3, 0] = max(12.0, float(self.kf.statePost[3, 0]))

        self.last_observed_bbox = np.array(bbox, dtype=np.float32).copy()
        self.last_bbox = self._state_to_bbox(self.kf.statePost)

    def update_optical_flow(self, dx: float, dy: float) -> None:
        """Apply bounded optical-flow translation to coasting tracks without mutating box shape."""
        self.kf.statePost[0, 0] += float(dx)
        self.kf.statePost[1, 0] += float(dy)
        self.last_bbox[0] += float(dx)
        self.last_bbox[1] += float(dy)
        self.last_bbox[2] += float(dx)
        self.last_bbox[3] += float(dy)

    def get_bbox(self) -> np.ndarray:
        return self.last_bbox


def calculate_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two sets of bounding boxes [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.expand_dims(boxes_a, 1)
    boxes_b = np.expand_dims(boxes_b, 0)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_a = np.maximum(0.0, boxes_a[..., 2] - boxes_a[..., 0]) * np.maximum(
        0.0, boxes_a[..., 3] - boxes_a[..., 1]
    )
    area_b = np.maximum(0.0, boxes_b[..., 2] - boxes_b[..., 0]) * np.maximum(
        0.0, boxes_b[..., 3] - boxes_b[..., 1]
    )
    union = area_a + area_b - intersection

    return np.where(union > 0, intersection / union, 0.0)


class ByteTracker:
    """
    Event-isolated ByteTrack tracker with configurable lost-track memory buffer
    and inner-ROI outlier-rejected optical flow for coasting frames.
    """

    def __init__(
        self,
        fps: float = 25.0,
        lost_buffer_seconds: float = 3.0,
        match_thresh: float = 0.2,
    ):
        self.fps = max(float(fps), 1.0)
        self.lost_track_buffer = max(1, int(round(self.fps * lost_buffer_seconds)))
        self.match_thresh = float(match_thresh)
        self.tracked_stracks: List[KalmanBoxTracker] = []
        self.prev_gray: Optional[np.ndarray] = None
        KalmanBoxTracker.count = 0

    def step(
        self,
        frame_img: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        active_event_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        h_img, w_img = frame_img.shape[:2]

        for trk in self.tracked_stracks:
            trk.predict()

        unmatched_tracks: List[int] = list(range(len(self.tracked_stracks)))

        if detections is not None and len(detections) > 0:
            det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
            det_scores = np.array(
                [d.get("score", 1.0) for d in detections], dtype=np.float32
            )
            det_labels = [str(d.get("label", "target")) for d in detections]
            det_event_ids = [int(d.get("event_id", 0) or 0) for d in detections]

            high_thresh = 0.4
            high_mask = det_scores >= high_thresh
            first_det_boxes = det_boxes[high_mask]
            first_det_scores = det_scores[high_mask]
            first_det_labels = [det_labels[i] for i, m in enumerate(high_mask) if m]
            first_det_eids = [det_event_ids[i] for i, m in enumerate(high_mask) if m]

            second_det_boxes = det_boxes[~high_mask]
            second_det_scores = det_scores[~high_mask]
            second_det_labels = [det_labels[i] for i, m in enumerate(~high_mask) if m]
            second_det_eids = [det_event_ids[i] for i, m in enumerate(~high_mask) if m]

            (
                matched_tracks,
                unmatched_tracks,
                unmatched_dets,
            ) = self._associate(
                self.tracked_stracks,
                first_det_boxes,
                first_det_eids,
                self.match_thresh,
                max_center_dist=0.16 * w_img,
            )

            for t_idx, d_idx in matched_tracks:
                self.tracked_stracks[t_idx].update(
                    first_det_boxes[d_idx],
                    float(first_det_scores[d_idx]),
                    first_det_labels[d_idx],
                    first_det_eids[d_idx],
                )

            if unmatched_tracks and len(second_det_boxes) > 0:
                rem_tracks = [self.tracked_stracks[i] for i in unmatched_tracks]
                matched_sec, unmatched_sec_rel, _ = self._associate(
                    rem_tracks,
                    second_det_boxes,
                    second_det_eids,
                    max(0.05, self.match_thresh - 0.1),
                    max_center_dist=0.20 * w_img,
                )
                for t_rel_idx, d_idx in matched_sec:
                    real_t_idx = unmatched_tracks[t_rel_idx]
                    self.tracked_stracks[real_t_idx].update(
                        second_det_boxes[d_idx],
                        float(second_det_scores[d_idx]),
                        second_det_labels[d_idx],
                        second_det_eids[d_idx],
                    )
                unmatched_tracks = [unmatched_tracks[i] for i in unmatched_sec_rel]

            for t_idx in unmatched_tracks:
                self.tracked_stracks[t_idx].is_lost = True

            for d_idx in unmatched_dets:
                self.tracked_stracks.append(
                    KalmanBoxTracker(
                        first_det_boxes[d_idx],
                        float(first_det_scores[d_idx]),
                        first_det_labels[d_idx],
                        event_id=first_det_eids[d_idx],
                    )
                )
        else:
            for trk in self.tracked_stracks:
                trk.is_lost = True

        # Retain active-event lost tracks up to lost_track_buffer; retire finished events promptly.
        retained_tracks: List[KalmanBoxTracker] = []
        for trk in self.tracked_stracks:
            if trk.time_since_update > self.lost_track_buffer:
                continue
            if (
                active_event_ids is not None
                and trk.event_id > 0
                and trk.time_since_update > 0
                and trk.event_id not in active_event_ids
            ):
                continue
            retained_tracks.append(trk)
        self.tracked_stracks = retained_tracks

        # Apply optical-flow guidance ONLY when at least one active track is coasting.
        need_flow = any(trk.time_since_update > 0 for trk in self.tracked_stracks)
        if need_flow:
            gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                self._refine_with_optical_flow(self.prev_gray, gray)
            self.prev_gray = gray
        else:
            self.prev_gray = None

        results: List[Dict[str, Any]] = []
        for trk in self.tracked_stracks:
            b = trk.get_bbox()
            x1 = max(0.0, min(float(w_img), float(b[0])))
            y1 = max(0.0, min(float(h_img), float(b[1])))
            x2 = max(0.0, min(float(w_img), float(b[2])))
            y2 = max(0.0, min(float(h_img), float(b[3])))
            results.append(
                {
                    "track_id": trk.id,
                    "event_id": trk.event_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "score": round(float(trk.score), 2),
                    "label": trk.label,
                    "is_lost": trk.time_since_update > 0,
                    "time_lost_frames": trk.time_since_update,
                }
            )

        return results

    def _associate(
        self,
        tracks: List[KalmanBoxTracker],
        det_boxes: np.ndarray,
        det_event_ids: List[int],
        threshold: float,
        max_center_dist: float = 150.0,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or len(det_boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(det_boxes)))

        track_boxes = np.array([t.get_bbox() for t in tracks], dtype=np.float32)
        iou_matrix = calculate_iou_matrix(track_boxes, det_boxes)

        trk_centers = np.column_stack(
            (
                (track_boxes[:, 0] + track_boxes[:, 2]) / 2.0,
                (track_boxes[:, 1] + track_boxes[:, 3]) / 2.0,
            )
        )
        det_centers = np.column_stack(
            (
                (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0,
                (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0,
            )
        )
        dist_matrix = np.linalg.norm(
            trk_centers[:, None, :] - det_centers[None, :, :], axis=-1
        )

        area_trk = np.maximum(
            1.0,
            (track_boxes[:, 2] - track_boxes[:, 0])
            * (track_boxes[:, 3] - track_boxes[:, 1]),
        )
        area_det = np.maximum(
            1.0,
            (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1]),
        )
        area_ratio = area_trk[:, None] / area_det[None, :]

        cost_matrix = (1.0 - iou_matrix) + 0.3 * np.clip(
            dist_matrix / max(max_center_dist, 1.0), 0.0, 2.0
        )

        for r, trk in enumerate(tracks):
            for c, det_eid in enumerate(det_event_ids):
                if trk.event_id > 0 and det_eid > 0 and trk.event_id != det_eid:
                    cost_matrix[r, c] = 1e6

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(det_boxes)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= 1e5:
                continue
            size_compatible = 0.35 <= area_ratio[r, c] <= 2.85
            if iou_matrix[r, c] >= threshold or (
                dist_matrix[r, c] <= max_center_dist and size_compatible
            ):
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, sorted(unmatched_tracks), sorted(unmatched_dets)

    def _refine_with_optical_flow(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> None:
        h_img, w_img = prev_gray.shape[:2]
        # Scale per-frame displacement limit dynamically to FPS so fast motion works at lower FPS too.
        max_disp = min(24.0, max(5.0, (24.0 / self.fps) * 10.0))

        for trk in self.tracked_stracks:
            if trk.time_since_update <= 0 or trk.time_since_update > int(self.fps * 2.0):
                continue

            b = trk.get_bbox()
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            # Sample inner 60% core of the bounding box to avoid static floor and passerby legs at borders.
            x1 = max(0, int(b[0] + 0.20 * bw))
            y1 = max(0, int(b[1] + 0.20 * bh))
            x2 = min(w_img, int(b[2] - 0.20 * bw))
            y2 = min(h_img, int(b[3] - 0.20 * bh))
            if (x2 - x1) < 12 or (y2 - y1) < 12:
                continue

            mask = np.zeros_like(prev_gray)
            mask[y1:y2, x1:x2] = 255
            pts0 = cv2.goodFeaturesToTrack(
                prev_gray, mask=mask, maxCorners=20, qualityLevel=0.04, minDistance=4
            )
            if pts0 is None or len(pts0) < 4:
                continue

            pts1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts0, None, winSize=(15, 15), maxLevel=2
            )
            if pts1 is None or st is None:
                continue

            good0 = pts0[st == 1]
            good1 = pts1[st == 1]
            if len(good0) < 4:
                continue

            deltas = good1 - good0
            med_dx = float(np.median(deltas[:, 0]))
            med_dy = float(np.median(deltas[:, 1]))

            # Filter outliers (static floor / stray corners) relative to median motion.
            devs = np.hypot(deltas[:, 0] - med_dx, deltas[:, 1] - med_dy)
            inliers = deltas[devs <= 3.0]
            if len(inliers) < 3:
                continue

            dx = float(np.median(inliers[:, 0]))
            dy = float(np.median(inliers[:, 1]))
            mag = float(np.hypot(dx, dy))
            if 0.35 <= mag <= max_disp:
                trk.update_optical_flow(dx, dy)
