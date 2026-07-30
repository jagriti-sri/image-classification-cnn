
# Assignment-9.py
import os, random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path="PetImages"

print("Folder Structure:")
for f in os.listdir(dataset_path):
    print("-",f)

classes = sorted([
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])
total=0
for c in classes:
    total+=len(os.listdir(os.path.join(dataset_path,c)))
print("Total Images:",total)

plt.figure(figsize=(10,4))
count=1
for c in classes:
    imgs=[i for i in os.listdir(os.path.join(dataset_path,c)) if i.lower().endswith((".jpg",".jpeg",".png"))]
    for img in random.sample(imgs,min(3,len(imgs))):
        if count>5: break
        plt.subplot(1,5,count)
        plt.imshow(plt.imread(os.path.join(dataset_path,c,img)))
        plt.title(c); plt.axis("off"); count+=1
    if count>5: break
plt.tight_layout()
plt.savefig("Figure_1.png")
plt.close()
print("Creating ImageDataGenerator...")
gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading training data...")
train=gen.flow_from_directory(dataset_path,target_size=(128,128),batch_size=32,class_mode="binary",subset="training")
print("Training data loaded.")
test=gen.flow_from_directory(dataset_path,target_size=(128,128),batch_size=32,class_mode="binary",subset="validation",shuffle=False)
print("Validation data loaded.")
model=Sequential([
Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
MaxPooling2D((2,2)),
Conv2D(64,(3,3),activation="relu"),
MaxPooling2D((2,2)),
Conv2D(128,(3,3),activation="relu"),
MaxPooling2D((2,2)),
Flatten(),
Dense(128,activation="relu"),
Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
history=model.fit(train,validation_data=test,epochs=10)
print("Starting CNN training...")
loss,acc=model.evaluate(test)
print("Test Accuracy:",acc)

pred=(model.predict(test)>0.5).astype(int).flatten()
true=test.classes
print("Precision:",precision_score(true,pred))
print("Recall:",recall_score(true,pred))
print("F1:",f1_score(true,pred))
print(confusion_matrix(true,pred))
print(classification_report(true,pred))

plt.figure()
plt.plot(history.history["accuracy"],label="Train")
plt.plot(history.history["val_accuracy"],label="Validation")
plt.legend(); plt.title("Accuracy vs Epoch")
plt.savefig("Figure_2.png")
plt.close()
plt.figure()
plt.plot(history.history["loss"],label="Train")
plt.plot(history.history["val_loss"],label="Validation")
plt.legend(); plt.title("Loss vs Epoch")
plt.savefig("Figure_3.png")
plt.close()
print("Observations and Conclusion printed successfully.")
