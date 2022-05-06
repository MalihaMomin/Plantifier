# Plantifier
Plantifier is a website that identifies plant on the basis of its leaf structure. It is developed using machine learning and image processing.

# Overview
It identifies 32 different species of plants based on leaf image. This model used Support Vector Machine (SVM) classifier and was able to classify with 90% accuracy.

# Dataset
For this project, we used Flavia Leaves Dataset. <br>
Link to download the dataset: <br>
https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/

# Dependencies
1. numpy 
2. pandas 
3. os 
4. sklearn 
5. matplotlib 
6. opencv 
7. mahotas 
8. flask 
<br>
This code is executed on VScode

# Instructions
1. Create 2 folders inside the "**static**" folder and name it as, "**Leaves**" and "**prediction**". (names are **case sensitive**)
2. Download the dataset from the link provided.
3. Keep all the images of the dataset in the "**Leaves**" folder.
4. Install all the dependencies.

# Steps to Run this Project
1. Open the project folder in vscode.
2. Run "Plantifier.py" file.
3. A terminal window will open and the backend processing will start.
4. At the end, you will get to see some statements like "Debugger is active" or "Debugger is on" (It may take 25sec approx.)
5. After you see this statement in your terminal window, open any browser and in the address bar type, "localhost:5000"
6. Hit enter, and the website is up and running.
<br>
PS: It may take approx. 20sec to give the final output.
