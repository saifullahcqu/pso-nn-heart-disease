==============================
README - PSO Neural Network
==============================

Student Name: Saifullah 
Student ID   : 12250129  
Unit Code    : COIT29224  
Assessment   : Assignment 1 - Evolutionary Computation  

Project Title: Enhancing Neural Network Performance with PSO

------------------------------
About the Project
------------------------------
This project applies Particle Swarm Optimization (PSO) to improve the performance of a neural 
network used for heart disease prediction. The model uses a real dataset (Cleveland Heart Disease) 
and compares the results between a traditionally tuned neural network (GridSearchCV) and one 
optimized using PSO.

The goal is to show how PSO can effectively tune hyperparameters like learning rate and number 
of neurons to improve model accuracy.

------------------------------
What’s Included
------------------------------
1. **pso_nn_heart_disease.py**  
   -> Python code implementing baseline NN and PSO-optimized NN

2. **heart_cleveland_upload.csv**  
   -> Dataset used for training and testing

3. **output/** (auto-created)  
   -> Contains result graphs:
   - `pso_optimization_progress.png` — accuracy improvement over generations  
   - `model_comparison.png` — bar chart comparing baseline NN vs PSO-NN  

------------------------------
How to Run
------------------------------
1. Open terminal or command prompt
2. Navigate to the folder where files are saved:

   Example:
   > cd E:\PSO_NN_Project

3. (Optional) Activate virtual environment if using:
   > .\venv\Scripts\activate

4. Install required libraries:
   > pip install -r requirements.txt

5. Run the code:
   > python pso_nn_heart_disease.py

6. Check the `output/` folder for two graphs:
   - Accuracy trend across PSO generations  
   - Final comparison of both models

------------------------------
Requirements
------------------------------
Make sure the following Python libraries are installed:
- pandas
- numpy
- scikit-learn
- matplotlib

You can install them easily by running:
> pip install -r requirements.txt
