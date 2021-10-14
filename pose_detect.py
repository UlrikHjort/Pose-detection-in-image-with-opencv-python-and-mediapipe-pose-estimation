import cv2
import mediapipe as mp

image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pose_landmarks = mp.solutions.pose.Pose().process(image_rgb).pose_landmarks

mp.solutions.drawing_utils.draw_landmarks(image, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS) if pose_landmarks else exit(0)

cv2.imwrite("./pose.jpg",image)



