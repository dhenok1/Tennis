from ultralytics import YOLO
import cv2
import pickle
from utils import get_center_of_bbox, measure_distance

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        # Get the 2 players currntly in the match
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        # Loop over detections
        for player_dict in player_detections:
            # Filter each dictionary so that in contains only the detections of the 2 desired players
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            # Create sepreate detection list for them
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        # Loop through
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_dist = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_dist:
                    min_dist = distance
            distances.append((track_id, min_dist))

        # Sort in ascending order based on min dist
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 track id (the 2 players who are actually playing)
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        #Loop through and detect people each frame
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections
    
    def detect_frame(self, frame):
        # Tracks people in current frame and persists info between subsequent calls
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        #Loop through people identified
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            #Bounding box boundaries
            result = box.xyxy.tolist()[0]
            #Get class id and name
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            #Makes sure people are the objects identifid
            if object_cls_name == "person":
                player_dict[track_id] = result

            return player_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames 