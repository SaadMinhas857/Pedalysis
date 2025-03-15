import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker
        Args:
            max_age (int): Maximum number of frames to keep alive a track without associated detections
            min_hits (int): Minimum number of associated detections before track is initialized
            iou_threshold (float): Minimum IOU for match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_id = 0
        
    def _iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IOU between two boxes"""
        xx1 = max(bbox1[0], bbox2[0])
        yy1 = max(bbox1[1], bbox2[1])
        xx2 = min(bbox1[2], bbox2[2])
        yy2 = min(bbox1[3], bbox2[3])
        
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        intersection = w * h
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _associate_detections_to_trackers(self, detections: np.ndarray, trackers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Associate detections with tracked objects using IOU
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
                
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in row_ind:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in col_ind:
                unmatched_trackers.append(t)
                
        matches = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_detections.append(r)
                unmatched_trackers.append(c)
            else:
                matches.append([r, c])
                
        return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Update tracked objects with new detections
        Args:
            dets: nx5 numpy array of detections [x1,y1,x2,y2,conf]
        Returns:
            nx5 numpy array of tracked objects [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            try:
                pos = trk.predict()
                if pos is None:
                    to_del.append(t)
                    continue
                trks[t, :4] = pos
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            except:
                to_del.append(t)
        
        # Remove dead tracklets
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanTracker(dets[i, :])
            trk.id = self.track_id
            self.track_id += 1
            self.trackers.append(trk)
        
        # Get states of all trackers
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < self.max_age:
                d = trk.get_state()
                if d is not None:
                    ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

class KalmanTracker:
    """
    Kalman Filter tracker for vehicle motion
    """
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:,2:] *= 10
        self.kf.P[4:,4:] *= 1000
        self.kf.P *= 10
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _convert_bbox_to_z(self, bbox):
        """
        Convert bounding box to KF state vector [x,y,s,r]
        x,y is the center of the box
        s is the scale/area
        r is the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x):
        """
        Convert KF state vector [x,y,s,r] to bounding box [x1,y1,x2,y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).reshape((4,))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def get_state(self):
        """
        Returns the current bounding box estimate
        """
        return self._convert_x_to_bbox(self.kf.x) 