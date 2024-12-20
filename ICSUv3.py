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
from collections import defaultdict


class SceneUnderstanding:
    def __init__(self, pose_model_path, tool_model_path, video_path):
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
            label = self.tool_model.names[class_id]

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
        left_shoulder = keypoint[5][:2] if keypoint[5][2] > 0.5 else None
        left_hip = keypoint[11][:2] if keypoint[11][2] > 0.5 else None
        left_knee = keypoint[13][:2] if keypoint[13][2] > 0.5 else None

        framing = False
        assembling = False

        if left_shoulder is not None and left_hip is not None and left_knee is not None:
            angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            if angle < 100:
                framing = True
                assembling = True  
            elif angle < 120:
                assembling = True

        return framing, assembling

    def update_relationship_tracker(self, worker_id, relationship):

        self.relationship_tracker[worker_id][relationship] += 1

        for rel in list(self.relationship_tracker[worker_id].keys()):
            if rel != relationship:
                self.relationship_tracker[worker_id][rel] = 0

    def analyze_pose_with_graph(self, pose_results, tool_results, frame):
        self.graph.clear()
        worker_tool_status = []
        active_workers = set()
        nose_positions = {}

        for box in tool_results.boxes:
            xyxy = box.xyxy[0]
            x_min, y_min, x_max, y_max = map(int, xyxy)
            # tool_id = int(box.cls[0])
            # tool_node = f"tool{tool_id}"
            tool_name = self.tool_model.names[int(box.cls[0])]
            tool_node = f"{tool_name} (Tool {int(box.cls[0])})"

            self.graph.add_node(tool_node, pos=((x_min + x_max) // 2, (y_min + y_max) // 2))  
            self.draw_tool(frame, tool_results) 

        for result in pose_results:
            keypoints = result.keypoints.cpu().numpy()
            if result.boxes.id is None:
                continue
            worker_id = int(result.boxes.id.cpu().numpy())
            worker_node = f"worker{worker_id}"
            active_workers.add(worker_id)

            is_framing = False
            is_assembling = False
            held_tools = []

            for keypoint in keypoints.data:

                if keypoint[11, 2] > 0.5:  
                    hip_x, hip_y = int(keypoint[11, 0]), int(keypoint[11, 1])
                    self.graph.add_node(worker_node, pos=(hip_x, hip_y))
                    nose_positions[worker_id] = (hip_x, hip_y)

                x_coords = keypoint[:, 0][keypoint[:, 2] > 0.5]
                y_coords = keypoint[:, 1][keypoint[:, 2] > 0.5]
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    self.draw_worker(frame, keypoint, (x_min, y_min, x_max, y_max), worker_id)

                framing, assembling = self.analyze_posture(keypoint)
                if framing:
                    is_framing = True
                    self.graph.add_edge(worker_node, worker_node, relation="framing")
                    self.update_relationship_tracker(worker_id, "framing")
                if assembling:
                    is_assembling = True
                    self.graph.add_edge(worker_node, worker_node, relation="assembling")
                    self.update_relationship_tracker(worker_id, "assembling")

                    for i, (x, y, conf) in enumerate(keypoint):
                        if conf < 0.5 or i not in [9, 10]:  
                            continue
                        for box in tool_results.boxes:
                            x_min_tool, y_min_tool, x_max_tool, y_max_tool = map(int, box.xyxy[0])
                            if x_min_tool <= x <= x_max_tool and y_min_tool <= y <= y_max_tool:
                                tool_node = f"tool{int(box.cls[0])}"
                                held_tools.append(tool_node)
                                self.graph.add_edge(worker_node, tool_node, relation="holding")
                                self.update_relationship_tracker(worker_id, f"holding {tool_node}")
                                break
            if held_tools:
                if is_framing:
                    worker_tool_status.append(f"{worker_node} is holding tool and framing.")
                if not is_framing:
                    worker_tool_status.append(f"{worker_node} is holding tool.")
            else:
                if is_assembling:
                    worker_tool_status.append(f"{worker_node} is assembling materials.")
                if not is_assembling:
                    worker_tool_status.append(f"{worker_node} is not holding tool.")
            
            for relationship, count in self.relationship_tracker[worker_id].items():
                if count >= self.frame_threshold: 
                    worker_tool_status.append(f"{worker_node} is {relationship}.")            


        for i, id1 in enumerate(active_workers):
            for id2 in list(active_workers)[i + 1:]:
                pos1, pos2 = nose_positions[id1], nose_positions[id2]
                if self.calculate_distance(pos1, pos2) < self.distance_threshold:
                    worker_node1 = f"worker{id1}"
                    worker_node2 = f"worker{id2}"
                    self.graph.add_edge(worker_node1, worker_node2, relation="work with")
                    worker_tool_status.append(f"{worker_node1} is working with {worker_node2}.")

        # self.draw_graph(frame)

        return worker_tool_status

    

    def draw_graph(self, frame):

        pos = nx.get_node_attributes(self.graph, "pos")

        for edge in self.graph.edges(data=True):
            node1, node2, data = edge
            if node1 in pos and node2 in pos:
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                relation = data["relation"]

                line_type = cv2.LINE_4  
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, line_type)

                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(frame, relation, (mid_x - 20, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for node, (x, y) in pos.items():
            color = (0, 50, 255) if "worker" in node else (255, 0, 0) 
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.putText(frame, node, (x - 30, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    def scene_understanding(self, frame):
        pose_results, tool_results = self.load_models(frame)
        worker_tool_status = self.analyze_pose_with_graph(pose_results, tool_results, frame)

        for i, relationship in enumerate(worker_tool_status):

            text = str(relationship).encode("ascii", errors="ignore").decode("ascii")
            cv2.putText(frame, text, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 3)

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


scene = SceneUnderstanding("pose.pt", "best.pt", "5.mp4")
scene.run()
