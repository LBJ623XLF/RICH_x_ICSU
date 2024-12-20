# Copyright (c) 2024 [RICHEASYGOAT]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cv2
import numpy as np
import networkx as nx
from ultralytics import YOLO
from collections import defaultdict, deque

class SceneUnderstanding:
    def __init__(self, pose_model_path, tool_model_path, video_path, config_path):
        self.pose_model = YOLO(pose_model_path)
        self.tool_model = YOLO(tool_model_path)
        self.cap = cv2.VideoCapture(video_path)

        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7],
            [6, 8], [7, 9], [8, 10], [9, 11]
        ]
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)
        self.kpt_color = self.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.limb_color = self.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.timeout_frames = 60
        self.current_frame = 0
        self.distance_threshold = 5000
        self.graph = nx.DiGraph()
        self.relationship_tracker = defaultdict(lambda: defaultdict(int))
        self.frame_threshold = 10
        self.task_config = self.load_config(config_path)
        self.temporal_memory = defaultdict(lambda: deque(maxlen=5))

    def load_config(self, config_path):
        config = {}
        with open(config_path, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    config[key.strip()] = value.strip()

        config['keypoints_used'] = list(map(int, config['keypoints_used'].split(',')))
        config['angle_threshold'] = float(config['angle_threshold'])
        config['holding_tool'] = config['holding_tool'].lower() == 'yes'
        return config

    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def calculate_angle(self, a, b, c):
        v1 = np.array([a[0] - b[0], a[1] - b[1]])
        v2 = np.array([c[0] - b[0], c[1] - b[1]])
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return np.degrees(angle)

    def load_models(self, frame):
        pose_results = self.pose_model.track(frame, conf=0.7, persist=True, verbose=False)[0]
        tool_results = self.tool_model.track(frame, conf=0.7, persist=True, verbose=False)[0]
        return pose_results, tool_results

    def draw_tool(self, frame, tool_results):
        for box in tool_results.boxes:
            xyxy = box.xyxy[0]
            class_id = int(box.cls[0])
            label = self.tool_model.names[class_id] if class_id in self.tool_model.names else "Unknown Tool"

            x_min, y_min, x_max, y_max = map(int, xyxy)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            cv2.putText(frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), 2)

    def draw_worker(self, frame, keypoint, bbox, worker_id):
        x_min, y_min, x_max, y_max = bbox

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 50, 255), 3)
        cv2.putText(frame, f"Worker{worker_id}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        for i, (x, y, conf) in enumerate(keypoint):
            if conf < 0.5:
                continue
            color_k = [int(c) for c in self.kpt_color[i]]
            cv2.circle(frame, (int(x), int(y)), 8, color_k, -1, lineType=cv2.LINE_AA)

        for i, sk in enumerate(self.skeleton):
            pos1 = (int(keypoint[sk[0] - 1, 0]), int(keypoint[sk[0] - 1, 1]))
            pos2 = (int(keypoint[sk[1] - 1, 0]), int(keypoint[sk[1] - 1, 1]))
            conf1 = keypoint[sk[0] - 1, 2]
            conf2 = keypoint[sk[1] - 1, 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
            cv2.line(frame, pos1, pos2, [int(c) for c in self.limb_color[i]], thickness=3)

    def analyze_posture(self, keypoint):
        keypoints_indices = self.task_config['keypoints_used']
        points = [(keypoint[i][:2] if keypoint[i][2] > 0.5 else None) for i in keypoints_indices]

        if all(p is not None for p in points):
            angle = self.calculate_angle(points[0], points[1], points[2])
            if angle < self.task_config['angle_threshold']:
                return True
        return False

    def analyze_pose_with_graph(self, pose_results, tool_results, frame):
        self.graph.clear()
        worker_tool_status = []
        active_workers = set()
        nose_positions = {}

        for box in tool_results.boxes:
            xyxy = box.xyxy[0]
            x_min, y_min, x_max, y_max = map(int, xyxy)
            tool_node = self.tool_model.names[int(box.cls[0])] if int(box.cls[0]) in self.tool_model.names else "Unknown Tool"
            self.graph.add_node(tool_node, pos=((x_min + x_max) // 2, (y_min + y_max) // 2))
            self.draw_tool(frame, tool_results)

        for result in pose_results:
            keypoints = result.keypoints.cpu().numpy()
            if result.boxes.id is None:
                continue

            worker_id = int(result.boxes.id.cpu().numpy())
            worker_node = f"worker{worker_id}"
            active_workers.add(worker_id)
            is_holding_tool = False
            holding_tool_name = None

            for keypoint in keypoints.data:
                x_coords = keypoint[:, 0][keypoint[:, 2] > 0.5]
                y_coords = keypoint[:, 1][keypoint[:, 2] > 0.5]
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    self.draw_worker(frame, keypoint, (x_min, y_min, x_max, y_max), worker_id)

                if keypoint[0, 2] > 0.5:
                    nose_positions[worker_id] = (int(keypoint[0, 0]), int(keypoint[0, 1]))

                for box in tool_results.boxes:
                    x_min_tool, y_min_tool, x_max_tool, y_max_tool = map(int, box.xyxy[0])
                    if (keypoint[9][0] > x_min_tool and keypoint[9][0] < x_max_tool and
                        keypoint[9][1] > y_min_tool and keypoint[9][1] < y_max_tool) or \
                       (keypoint[10][0] > x_min_tool and keypoint[10][0] < x_max_tool and
                        keypoint[10][1] > y_min_tool and keypoint[10][1] < y_max_tool):
                        tool_node = self.tool_model.names[int(box.cls[0])] if int(box.cls[0]) in self.tool_model.names else "Unknown Tool"
                        if tool_node == self.task_config['tool_needed']:
                            self.graph.add_edge(worker_node, tool_node, relation="holding")
                            is_holding_tool = True
                            holding_tool_name = tool_node

                if is_holding_tool:
                    if self.analyze_posture(keypoint):
                        self.graph.add_edge(worker_node, worker_node, relation=self.task_config['task'])
                        status = f"{worker_node} is holding {holding_tool_name} and {self.task_config['task']}"
                        self.relationship_tracker[worker_id][status] += 1
                        if self.relationship_tracker[worker_id][status] >= self.frame_threshold:
                            self.temporal_memory[worker_node].append(status)
                    else:
                        status = f"{worker_node} is holding {holding_tool_name}"
                        self.relationship_tracker[worker_id][status] += 1
                        if self.relationship_tracker[worker_id][status] >= self.frame_threshold:
                            self.temporal_memory[worker_node].append(status)

        for i, id1 in enumerate(active_workers):
            for id2 in list(active_workers)[i + 1:]:
                pos1, pos2 = nose_positions.get(id1), nose_positions.get(id2)
                if pos1 and pos2 and self.calculate_distance(pos1, pos2) < self.distance_threshold:
                    worker_node1 = f"worker{id1}"
                    worker_node2 = f"worker{id2}"
                    status = f"{worker_node1} is working with {worker_node2}."
                    self.relationship_tracker[(worker_node1, worker_node2)][status] += 1
                    if self.relationship_tracker[(worker_node1, worker_node2)][status] >= self.frame_threshold:
                        self.temporal_memory[(worker_node1, worker_node2)].append(status)

        for worker_node, statuses in self.temporal_memory.items():
            if isinstance(worker_node, tuple):
                worker_tool_status.append(statuses[-1])
            else:
                worker_tool_status.append(statuses[-1])
        return worker_tool_status

    def scene_understanding(self, frame):
        pose_results, tool_results = self.load_models(frame)
        worker_tool_status = self.analyze_pose_with_graph(pose_results, tool_results, frame)

        for i, relationship in enumerate(worker_tool_status):
            cv2.putText(frame, relationship, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 3)

        return frame

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            self.current_frame += 1
            frame = self.scene_understanding(frame)

            cv2.imshow("Scene Understanding", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

scene = SceneUnderstanding("pose.pt", "best.pt", "5.mp4", "config.txt")
scene.run()


