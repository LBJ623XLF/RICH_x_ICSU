##VERSION0
# from ultralytics import YOLO
# import cv2
# import numpy as np

# model1 = YOLO('yolov8n-pose.pt')
# #model2 = YOLO('best.pt')
# model2 = YOLO('yolov8n.pt')

# #results = model(source="1.mp4", show=True, conf=0.4, save=True)
# #results = model(source="https://youtu.be/FLE0atU6LkU?si=86VidgZcAXWA8cms", show=True)

# video_path = "1.mp4"
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():

#     success, frame = cap.read()

#     if success:

#         results1 = model1(frame)[0]
#         results2 = model2(frame)[0]

#         skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#                     [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        
      
#         pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                                 [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                                 [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                                 [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
        
       
#         kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
#         limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


        
#         for result in results1:

#             keypoints = result.keypoints.cpu().numpy()

#             for keypoint in keypoints.data:
#                 for i, (x, y, conf) in enumerate(keypoint):
#                     color_k = [int(x) for x in kpt_color[i]]  
#                     if conf < 0.5:
#                         continue
#                     if x != 0 and y != 0:
#                         cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)  
#                 for i, sk in enumerate(skeleton):
#                     pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))  
#                     pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))  
#                     conf1 = keypoint[(sk[0] - 1), 2]  
#                     conf2 = keypoint[(sk[1] - 1), 2]  
#                     if conf1 < 0.5 or conf2 < 0.5:
#                         continue
#                     if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
#                         continue
#                     cv2.line(frame, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA) 


#         for box in results2.boxes:

#             xyxy = box.xyxy[0]
#             conf = box.conf[0]
#             class_id = int(box.cls[0])
#             label = model2.names[class_id]

#             x_min, y_min, x_max, y_max = map(int, xyxy)
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)

#             label_text = f"{lfrom ultralytics import YOLO


# import cv2
# import numpy as np

# # Load models
# model1 = YOLO('yolov8n-pose.pt')  # Pose estimation model
# model2 = YOLO('best.pt')       # Object detection model

# # Open video file
# video_path = "3.mp4"
# cap = cv2.VideoCapture(video_path)

# # Skeleton connections for YOLO pose model
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# # Define pose palette
# pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                          [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                          [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                          [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

# kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # Run pose estimation and object detection
#     results1 = model1(frame)[0]
#     results2 = model2(frame)[0]

#     worker_count = 0  # Counter for worker numbering

#     # Draw pose skeletons and label workers
#     for result in results1:
#         keypoints = result.keypoints.cpu().numpy()

#         for keypoint in keypoints.data:
#             worker_count += 1
#             worker_label = f"Worker{worker_count} not working"

#             for i, (x, y, conf) in enumerate(keypoint):
#                 color_k = [int(c) for c in kpt_color[i]]
#                 if conf < 0.5:  # Skip low-confidence keypoints
#                     continue
#                 if x != 0 and y != 0:
#                     cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

#                 # Check if the keypoint (e.g., hand) is within any bounding box
#                 for box in results2.boxes:
#                     x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
#                     if x_min <= x <= x_max and y_min <= y <= y_max:
#                         worker_label = f"Worker{worker_count} working"  # Update label if condition is met

#             for i, sk in enumerate(skeleton):
#                 pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
#                 pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
#                 conf1 = keypoint[(sk[0] - 1), 2]
#                 conf2 = keypoint[(sk[1] - 1), 2]
#                 if conf1 < 0.5 or conf2 < 0.5:
#                     continue
#                 if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
#                     continue
#                 cv2.line(frame, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

#             # Display worker label
#             cv2.putText(frame, worker_label, (10, 50 * worker_count), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
#                         (0, 255, 0), 2, lineType=cv2.LINE_AA)

#     # Draw object detection boxes
#     for box in results2.boxes:
#         xyxy = box.xyxy[0]
#         class_id = int(box.cls[0])
#         label = model2.names[class_id]

#         x_min, y_min, x_max, y_max = map(int, xyxy)
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

#         label_text = f"{label}"
#         text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
#         text_x = x_min
#         text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
#         cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
#         cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     # Display results
#     cv2.imshow("YOLO Inf", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:

#         break

# cap.release()
# cv2.destroyAllWindows()






# #VERSION1
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load models
# model1 = YOLO('yolov8n-pose.pt')  # Pose estimation model
# model2 = YOLO('best.pt')       # Object detection model

# # Open video file
# video_path = "3.mp4"
# cap = cv2.VideoCapture(video_path)

# # Skeleton connections for YOLO pose model
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# # Define pose palette
# pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                          [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                          [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                          [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

# kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # Run pose estimation and object detection
#     results1 = model1(frame)[0]
#     results2 = model2(frame)[0]

#     worker_count = 0  # Counter for worker numbering

#     # Draw pose skeletons and label workers
#     for result in results1:
#         keypoints = result.keypoints.cpu().numpy()

#         for keypoint in keypoints.data:
#             worker_count += 1
#             worker_label = f"Worker{worker_count} not working"

#             for i, (x, y, conf) in enumerate(keypoint):
#                 color_k = [int(c) for c in kpt_color[i]]
#                 if conf < 0.5:  # Skip low-confidence keypoints
#                     continue
#                 if x != 0 and y != 0:
#                     cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

#                 # Check if the keypoint (e.g., hand) is within any bounding box
#                 for box in results2.boxes:
#                     x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
#                     if x_min <= x <= x_max and y_min <= y <= y_max:
#                         worker_label = f"Worker{worker_count} working"  # Update label if condition is met

#             for i, sk in enumerate(skeleton):
#                 pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
#                 pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
#                 conf1 = keypoint[(sk[0] - 1), 2]
#                 conf2 = keypoint[(sk[1] - 1), 2]
#                 if conf1 < 0.5 or conf2 < 0.5:
#                     continue
#                 if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
#                     continue
#                 cv2.line(frame, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

#             # Display worker label
#             cv2.putText(frame, worker_label, (10, 50 * worker_count), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
#                         (0, 255, 0), 2, lineType=cv2.LINE_AA)

#     # Draw object detection boxes
#     for box in results2.boxes:
#         xyxy = box.xyxy[0]
#         class_id = int(box.cls[0])
#         label = model2.names[class_id]

#         x_min, y_min, x_max, y_max = map(int, xyxy)
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

#         label_text = f"{label}"
#         text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
#         text_x = x_min
#         text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
#         cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
#         cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     # Display results
#     cv2.imshow("YOLO Inf", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



#VERSION2
from ultralytics import YOLO
import cv2
import numpy as np

# Load models
model1 = YOLO('yolov8n-pose.pt')  # Pose estimation model
model2 = YOLO('best.pt')          # Object detection model

# Open video file
video_path = "4.mp4"
cap = cv2.VideoCapture(video_path)

# Skeleton connections for YOLO pose model
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# Define pose palette
pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run pose estimation and object detection
    results1 = model1(frame)[0]
    results2 = model2(frame)[0]

    worker_count = 0  # Counter for worker numbering
    worker_tool_status = []  # To hold status of each worker

    # Draw pose skeletons and label workers
    for result in results1:
        keypoints = result.keypoints.cpu().numpy()

        for keypoint in keypoints.data:
            worker_count += 1
            holding_tool = False  # Flag to check if the worker is holding a tool

            # Calculate bounding box for keypoints
            x_coords = keypoint[:, 0][keypoint[:, 2] > 0.5]
            y_coords = keypoint[:, 1][keypoint[:, 2] > 0.5]

            if len(x_coords) > 0 and len(y_coords) > 0:
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

                # Draw bounding box for the worker
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Label the worker
                worker_label = f"Worker{worker_count}"
                cv2.putText(frame, worker_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 2, lineType=cv2.LINE_AA)

            # Check if hand keypoints are in any tool detection bounding box
            for i, (x, y, conf) in enumerate(keypoint):
                if conf < 0.5 or (i not in [9, 10]):  # Check only left (9) and right (10) hand keypoints
                    continue
                for box in results2.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        holding_tool = True
                        break  # Stop checking other boxes once a match is found

            # Store worker's status
            status = f"{worker_label} holding tool" if holding_tool else f"{worker_label} not holding tool"
            worker_tool_status.append(status)

            # Draw keypoints and skeleton
            for i, (x, y, conf) in enumerate(keypoint):
                if conf < 0.5:
                    continue
                color_k = [int(c) for c in kpt_color[i]]
                cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
                pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
                conf1 = keypoint[(sk[0] - 1), 2]
                conf2 = keypoint[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                cv2.line(frame, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

    # Write overall status in the top-left corner
    for i, status in enumerate(worker_tool_status):
        cv2.putText(frame, status, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw object detection boxes
    for box in results2.boxes:
        xyxy = box.xyxy[0]
        class_id = int(box.cls[0])
        label = model2.names[class_id]

        x_min, y_min, x_max, y_max = map(int, xyxy)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        label_text = f"{label}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x_min
        text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display results
    cv2.imshow("YOLO Inf", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




# #VERSION2
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load models
# model1 = YOLO('yolov8n-pose.pt')  # Pose estimation model
# model2 = YOLO('best.pt')          # Object detection model

# # Open video file
# video_path = "4.mp4"
# cap = cv2.VideoCapture(video_path)

# # Skeleton connections for YOLO pose model
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# # Define pose palette
# pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                          [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                          [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                          [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

# kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

# # Tracker dictionary to hold unique worker IDs
# worker_tracker = {}
# next_worker_id = 1


# def calculate_iou(box1, box2):

#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2

#     # Calculate intersection coordinates
#     inter_x1 = max(x1, x3)
#     inter_y1 = max(y1, y3)
#     inter_x2 = min(x2, x4)
#     inter_y2 = min(y2, y4)

#     # Calculate intersection area
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

#     # Calculate union area
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x4 - x3) * (y4 - y3)
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area > 0 else 0


# while cap.isOpened():

#     success, frame = cap.read()
#     if not success:
#         break

#     # Run pose estimation
#     results1 = model1(frame)[0]
#     results2 = model2(frame)[0]

#     current_boxes = []  # To hold bounding boxes of workers detected in the current frame
#     active_worker_ids = []  # To hold IDs of workers present in the current frame
#     worker_tool_status = []  # To hold tool-holding status of each worker

#     for result in results1:
#         keypoints = result.keypoints.cpu().numpy()

#         for keypoint in keypoints.data:
#             holding_tool = False  # Flag to check if the worker is holding a tool

#             # Calculate bounding box for keypoints
#             x_coords = keypoint[:, 0][keypoint[:, 2] > 0.5]
#             y_coords = keypoint[:, 1][keypoint[:, 2] > 0.5]

#             if len(x_coords) > 0 and len(y_coords) > 0:
#                 x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
#                 y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
#                 current_boxes.append((x_min, y_min, x_max, y_max))

#                 # Assign an ID using IoU-based tracking
#                 assigned_id = None
#                 for worker_id, prev_box in worker_tracker.items():
#                     if calculate_iou(prev_box, (x_min, y_min, x_max, y_max)) > 0.5:
#                         assigned_id = worker_id
#                         worker_tracker[worker_id] = (x_min, y_min, x_max, y_max)
#                         break
#                 if assigned_id is None:
#                     assigned_id = next_worker_id
#                     worker_tracker[assigned_id] = (x_min, y_min, x_max, y_max)
#                     next_worker_id += 1

#                 # Add active worker ID for this frame
#                 active_worker_ids.append(assigned_id)

#                 # Check if hand keypoints are in any tool detection bounding box
#                 for i, (x, y, conf) in enumerate(keypoint):
#                     if conf < 0.5 or (i not in [9, 10]):  # Check only left (9) and right (10) hand keypoints
#                         continue
#                     for box in results2.boxes:
#                         x_min_tool, y_min_tool, x_max_tool, y_max_tool = map(int, box.xyxy[0])
#                         if x_min_tool <= x <= x_max_tool and y_min_tool <= y <= y_max_tool:
#                             holding_tool = True
#                             break  # Stop checking other boxes once a match is found

#                 # Store worker's status
#                 status = f"Worker{assigned_id} holding tool" if holding_tool else f"Worker{assigned_id} not holding tool"
#                 worker_tool_status.append(status)

#                 # Draw bounding box and label
#                 worker_label = f"Worker{assigned_id}"
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
#                 cv2.putText(frame, worker_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                             (0, 255, 0), 2, lineType=cv2.LINE_AA)

#                 # Draw keypoints and skeleton
#                 for i, (x, y, conf) in enumerate(keypoint):
#                     if conf < 0.5:
#                         continue
#                     color_k = [int(c) for c in kpt_color[i]]
#                     cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

#                 for i, sk in enumerate(skeleton):
#                     pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
#                     pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
#                     conf1 = keypoint[(sk[0] - 1), 2]
#                     conf2 = keypoint[(sk[1] - 1), 2]
#                     if conf1 < 0.5 or conf2 < 0.5:
#                         continue
#                     cv2.line(frame, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

                
#     for box in results2.boxes:
#         xyxy = box.xyxy[0]  # Bounding box coordinates [x_min, y_min, x_max, y_max]
#         class_id = int(box.cls[0])  # Detected class ID
#         label = model2.names[class_id]  # Class label using model class names

#         x_min, y_min, x_max, y_max = map(int, xyxy)
#         # Draw bounding box
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

#         # Add label above the box
#         label_text = f"{label}"
#         text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
#         text_x = x_min
#         text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
#         cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
#         cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     # Remove workers not in the current frame
#     worker_tracker = {worker_id: bbox for worker_id, bbox in worker_tracker.items() if worker_id in active_worker_ids}

#     # Display worker statuses in the top-left corner
#     for i, status in enumerate(worker_tool_status):
#         cv2.putText(frame, status, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#     # Display results
#     cv2.imshow("YOLO Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# #VERSION3
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load models
# model1 = YOLO('yolov8n-pose.pt')  # Pose estimation model
# model2 = YOLO('best.pt')          # Object detection model

# # Open video file
# video_path = "4.mp4"
# cap = cv2.VideoCapture(video_path)

# # Skeleton connections for YOLO pose model
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# # Define pose palette
# pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                          [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                          [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                          [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

# kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

# # Tracker dictionary to hold unique worker IDs and last seen frame
# worker_tracker = {}  # Format: {worker_id: {"bbox": (x_min, y_min, x_max, y_max), "last_seen": frame_number}}
# next_worker_id = 1
# iou_threshold = 0.3  # IoU threshold for matching detections
# frame_counter = 0  # Keep track of the current frame number
# max_disappear_frames = 5  # Remove workers not detected after these many frames


# def calculate_iou(box1, box2):
#     """Calculate Intersection over Union (IoU) between two bounding boxes."""
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2

#     # Calculate intersection coordinates
#     inter_x1 = max(x1, x3)
#     inter_y1 = max(y1, y3)
#     inter_x2 = min(x2, x4)
#     inter_y2 = min(y2, y4)

#     # Calculate intersection area
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

#     # Calculate union area
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x4 - x3) * (y4 - y3)
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area > 0 else 0


# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     frame_counter += 1
#     results1 = model1(frame)[0]
#     results2 = model2(frame)[0]

#     current_boxes = []  # To hold bounding boxes of workers detected in the current frame
#     active_worker_ids = []  # To hold IDs of workers present in the current frame
#     worker_tool_status = []  # To hold tool-holding status of each worker

#     for result in results1:
#         keypoints = result.keypoints.cpu().numpy()

#         for keypoint in keypoints.data:
#             holding_tool = False  # Flag to check if the worker is holding a tool

#             # Calculate bounding box for keypoints
#             x_coords = keypoint[:, 0][keypoint[:, 2] > 0.5]
#             y_coords = keypoint[:, 1][keypoint[:, 2] > 0.5]

#             if len(x_coords) > 0 and len(y_coords) > 0:
#                 x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
#                 y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
#                 current_boxes.append((x_min, y_min, x_max, y_max))

#                 # Assign an ID using IoU-based tracking
#                 assigned_id = None
#                 for worker_id, data in worker_tracker.items():
#                     if calculate_iou(data["bbox"], (x_min, y_min, x_max, y_max)) > iou_threshold:
#                         assigned_id = worker_id
#                         worker_tracker[worker_id] = {"bbox": (x_min, y_min, x_max, y_max), "last_seen": frame_counter}
#                         break
#                 if assigned_id is None:
#                     assigned_id = next_worker_id
#                     worker_tracker[assigned_id] = {"bbox": (x_min, y_min, x_max, y_max), "last_seen": frame_counter}
#                     next_worker_id += 1

#                 # Add active worker ID for this frame
#                 active_worker_ids.append(assigned_id)

#                 # Check if hand keypoints are in any tool detection bounding box
#                 for i, (x, y, conf) in enumerate(keypoint):
#                     if conf < 0.5 or (i not in [9, 10]):  # Check only left (9) and right (10) hand keypoints
#                         continue
#                     for box in results2.boxes:
#                         x_min_tool, y_min_tool, x_max_tool, y_max_tool = map(int, box.xyxy[0])
#                         if x_min_tool <= x <= x_max_tool and y_min_tool <= y <= y_max_tool:
#                             holding_tool = True
#                             break

#                 # Store worker's status
#                 status = f"Worker{assigned_id} holding tool" if holding_tool else f"Worker{assigned_id} not holding tool"
#                 worker_tool_status.append(status)

#                 # Draw bounding box and label
#                 worker_label = f"Worker{assigned_id}"
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
#                 cv2.putText(frame, worker_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                             (0, 255, 0), 2, lineType=cv2.LINE_AA)

#                 # Draw keypoints and skeleton
#                 for i, (x, y, conf) in enumerate(keypoint):
#                     if conf < 0.5:
#                         continue
#                     color_k = [int(c) for c in kpt_color[i]]
#                     cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

#                 for i, sk in enumerate(skeleton):
#                     pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
#                     pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
#                     conf1 = keypoint[(sk[0] - 1), 2]
#                     conf2 = keypoint[(sk[1] - 1), 2]
#                     if conf1 < 0.5 or conf2 < 0.5:
#                         continue
#                     cv2.line(frame, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

#     # Remove workers not seen recently
#     worker_tracker = {
#         worker_id: data for worker_id, data in worker_tracker.items()
#         if frame_counter - data["last_seen"] <= max_disappear_frames
#     }

#     # Display worker statuses in the top-left corner
#     for i, status in enumerate(worker_tool_status):
#         cv2.putText(frame, status, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#     # Display results
#     cv2.imshow("YOLO Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



# #VERSION4
# from ultralytics import YOLO
# import cv2
# import numpy as np


# model1 = YOLO('yolov8n-pose.pt')  
# model2 = YOLO('best.pt')          

# video_path = "4.mp4"
# cap = cv2.VideoCapture(video_path)

# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                          [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                          [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                          [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

# kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


# worker_tracker = {} 
# next_worker_id = 1
# iou_threshold = 0.3  
# frame_counter = 0 
# max_disappear_frames = 5  


# def calculate_iou(box1, box2):
    
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2


#     inter_x1 = max(x1, x3)
#     inter_y1 = max(y1, y3)
#     inter_x2 = min(x2, x4)
#     inter_y2 = min(y2, y4)


#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)


#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x4 - x3) * (y4 - y3)
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area > 0 else 0


# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     frame_counter += 1
#     results1 = model1(frame)[0]
#     results2 = model2(frame)[0]

#     current_boxes = []  
#     active_worker_ids = []  
#     worker_tool_status = [] 

#     for result in results1:
#         keypoints = result.keypoints.cpu().numpy()

#         for keypoint in keypoints.data:
#             holding_tool = False  


#             x_coords = keypoint[:, 0][keypoint[:, 2] > 0.5]
#             y_coords = keypoint[:, 1][keypoint[:, 2] > 0.5]

#             if len(x_coords) > 0 and len(y_coords) > 0:
#                 x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
#                 y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
#                 current_boxes.append((x_min, y_min, x_max, y_max))


#                 assigned_id = None
#                 for worker_id, data in worker_tracker.items():
#                     if calculate_iou(data["bbox"], (x_min, y_min, x_max, y_max)) > iou_threshold:
#                         assigned_id = worker_id
#                         worker_tracker[worker_id] = {"bbox": (x_min, y_min, x_max, y_max), "last_seen": frame_counter}
#                         break
#                 if assigned_id is None:
#                     assigned_id = next_worker_id
#                     worker_tracker[assigned_id] = {"bbox": (x_min, y_min, x_max, y_max), "last_seen": frame_counter}
#                     next_worker_id += 1


#                 active_worker_ids.append(assigned_id)


#                 for i, (x, y, conf) in enumerate(keypoint):
#                     if conf < 0.5 or (i not in [9, 10]):  
#                         continue
#                     for box in results2.boxes:
#                         x_min_tool, y_min_tool, x_max_tool, y_max_tool = map(int, box.xyxy[0])
#                         if x_min_tool <= x <= x_max_tool and y_min_tool <= y <= y_max_tool:
#                             holding_tool = True
#                             break

#                 status = f"Worker{assigned_id} holding tool" if holding_tool else f"Worker{assigned_id} not holding tool"
#                 worker_tool_status.append(status)

#                 worker_label = f"Worker{assigned_id}"
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
#                 cv2.putText(frame, worker_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                             (0, 255, 0), 2, lineType=cv2.LINE_AA)

#                 for i, (x, y, conf) in enumerate(keypoint):
#                     if conf < 0.5:
#                         continue
#                     color_k = [int(c) for c in kpt_color[i]]
#                     cv2.circle(frame, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

#                 for i, sk in enumerate(skeleton):
#                     pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
#                     pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
#                     conf1 = keypoint[(sk[0] - 1), 2]
#                     conf2 = keypoint[(sk[1] - 1), 2]
#                     if conf1 < 0.5 or conf2 < 0.5:
#                         continue
#                     cv2.line(frame, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

#     worker_tracker = {
#         worker_id: data for worker_id, data in worker_tracker.items()
#         if frame_counter - data["last_seen"] <= max_disappear_frames
#     }

#     for i, status in enumerate(worker_tool_status):
#         cv2.putText(frame, status, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#     cv2.imshow("YOLO Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
