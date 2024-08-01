This project will designing, building, testing and critiquing a system for Performing face alignment and CNN will used to landmarks and find mouth, nose in the test image.   Introduction

There have few steps to build up this system. 
Step 1 : Convert the image to grayscale, the single channel will more easily to training. 
Step 2 : Save the training data (training image.npz) with csv , the image part will be the  
training images and the keypoint part will be the label. 
Step 3 : Build up a CNN with five layers to make the prediction more accurately. 
Step 4 : Build a model and training data for better accuracy rate. 
Step 5: Prediction the test image and save the data with csv. 
Step 6: Find the mouth, nose coordinates and change it’s color.

![image](https://github.com/user-attachments/assets/59c70926-3b12-4d16-8ffc-0a17d436b9dc)


These are the example image face alignment result and only two points of left side of hand is 
not concrete. But this is only an ideal condition. Cause this face is just in the middle of the 
picture. 
No matter the skin color, the face alignment is similar. I guess cause every image will 

<img width="322" alt="image" src="https://github.com/user-attachments/assets/6b93cef0-6e0f-4dea-8cfa-0cacd318b571">

<img width="318" alt="image" src="https://github.com/user-attachments/assets/261dea96-29d8-4737-8f5a-6d5d65de434a">

<img width="318" alt="image" src="https://github.com/user-attachments/assets/f5593b56-cd93-43f6-ace3-c226ff11f679">


