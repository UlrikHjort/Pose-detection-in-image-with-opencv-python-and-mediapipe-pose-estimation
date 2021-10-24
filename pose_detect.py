#####################################################################-                                                                                                                                             
#         Image pose detection  with opencv and mediapipe                                                                                                                                                                
#                                                                                                                                                                                                                  
#           Copyright (C) 2021 By Ulrik Hørlyk Hjort                                                                                                                                                               
#                                                                                                                                                                                                                  
#  This Program is Free Software; You Can Redistribute It and/or                                                                                                                                                   
#  Modify It Under The Terms of The GNU General Public License                                                                                                                                                     
#  As Published By The Free Software Foundation; Either Version 2                                                                                                                                                  
#  of The License, or (at Your Option) Any Later Version.                                                                                                                                                          
#                                                                                                                                                                                                                  
#  This Program is Distributed in The Hope That It Will Be Useful,                                                                                                                                                 
#  But WITHOUT ANY WARRANTY; Without Even The Implied Warranty of                                                                                                                                                  
#  MERCHANTABILITY or FITNESS for A PARTICULAR PURPOSE.  See The                                                                                                                                                   
#  GNU General Public License for More Details.                                                                                                                                                                    
#                                                                                                                                                                                                                  
# You Should Have Received A Copy of The GNU General Public License                                                                                                                                                
# Along with This Program; if not, See <Http://Www.Gnu.Org/Licenses/>.                                                                                                                                             
#######################################################################-                                          
import cv2
import mediapipe as mp

image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pose_landmarks = mp.solutions.pose.Pose().process(image_rgb).pose_landmarks

mp.solutions.drawing_utils.draw_landmarks(image, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS) if pose_landmarks else exit(0)

cv2.imwrite("./pose.jpg",image)



