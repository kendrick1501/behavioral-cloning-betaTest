# Self-Driving Car Engineer Nanodegree

## Behavioral Cloning Project - Beta Test

**Name: Kendrick Amezquita**

**Email: kendrickamezquita@gmail.com**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

# Review

## Lessons

1. Lesson 1 chapter 6. link https://classroom.udacity.com/nanodegrees/nd026/parts/d0d74452-5fab-4c91-b417-afdc6048ccec/modules/b9184e78-b0bf-4725-8f78-fda88428a6df/lessons/facdff19-61a5-47e4-8179-a6a5dc28987f/concepts/undefined (JUPYTER in workspaces) not working

2. Lesson 1 chapter 6. The grader's output is "Oops, looks like you got an error!" despite runing the notebook file without warnings or errors. The outcome of the implemented model is attached below to further validate the correctness of the architecture.

Train on 80 samples, validate on 20 samples
```
Epoch 1/3
80/80 [==============================] - 0s 2ms/step - loss: 1.2441 - acc: 0.3875 - val_loss: 0.6926 - val_acc: 0.6500
Epoch 2/3
80/80 [==============================] - 0s 553us/step - loss: 0.7493 - acc: 0.6125 - val_loss: 0.5818 - val_acc: 0.8000
Epoch 3/3
80/80 [==============================] - 0s 642us/step - loss: 0.6166 - acc: 0.7750 - val_loss: 0.4226 - val_acc: 0.8000
```
It seems the grader only validates the outcome given by the architecture implemented as:

```
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
```
whereas for the alternative implementation:
```
#1st layer: flatten input layer
model.add(Flatten(input_shape=(32,32,3)))

#2nd layer: fully connected layer with relu activation function
model.add(Dense(128, activation = 'relu'))

#Output layer: fully connected layer with 5 classes
num_classes = 5
model.add(Dense(num_classes, activation='softmax'))
```
The output is not valid in spite of the accuracy of the model (>50%)

## Project

3. Project: Behavioral Cloning chapter 18. The generator quiz is solved already.

4. Project: Behavioral Cloning Workspace. Enabling disabling GPU takes a lot of time

5. Project: Behavioral Cloning Workspace. When opening the simulator's environment an error (No session for pid 57) is shown

6. Project: Behavioral Cloning Workspace. Error when opening terminator: "ln: failed to create symbolic link 'CarND-Behavioral-Cloning-P3/data': No such file or directory"

7. Project: Behavioral Cloning Workspace. Could not install dropbox service to save the collected data in the cloud

8. Project: Behavioral Cloning Workspace Simulator. There exists an important delay when driving the vehicle. This delay makes it difficult to drive safely, smoothly, and at the center of the lane. The situation is even more critical in track 2.

8. Project: Behavioral Cloning Workspace Simulator. The simulator freezes and jumps frames from time to time

9. Project: Behavioral Cloning Workspace Simulator. At some points of the tracks, when off-track, the car might get stuck even without obstacles around the vehicle

10. Project: Behavioral Cloning Workspace Simulator. I tested the (locally) trained model and the vehicle was able to complete the first track safely. However, the simulator didn't run smoothly. It was "flickering" during the simulation. Despite this performance, the output video "model_eval.mp4" seems to be smooth enough.

