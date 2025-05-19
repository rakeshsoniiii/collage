import cv2
import numpy as np
import os

data_path = 'faces/'
face_data = []
labels = []
label_dict = {}  # maps label number to username
current_label = 0

for user in os.listdir(data_path):
    user_path = os.path.join(data_path, user)
    if not os.path.isdir(user_path):
        continue

    label_dict[current_label] = user

    for img_file in os.listdir(user_path):
        image_path = os.path.join(user_path, img_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        face_data.append(np.asarray(img, dtype=np.uint8))
        labels.append(current_label)
    current_label += 1

labels = np.asarray(labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(face_data), labels)
model.save("face_model.xml")

# Save label dictionary
import pickle
with open("labels.pkl", "wb") as f:
    pickle.dump(label_dict, f)

print("Training complete for users:", label_dict)
