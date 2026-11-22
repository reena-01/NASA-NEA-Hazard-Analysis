# NASA Near-Earth Asteroid Hazard Analysis

## Python & Machine Learning Portfolio Project

A Python-based scientific data analysis and machine learning project
for analyzing Near-Earth Objects (NEOs) and predicting the NASA
potentially hazardous asteroid classification.

This project demonstrates a complete end-to-end workflow including:

- NASA API data acquisition
- Data loading and processing
- Data cleaning
- Feature engineering
- Exploratory data analysis
- Data visualization
- Feature selection
- Machine learning
- Model comparison
- Cross-validation
- Model evaluation
- Model persistence
- Prediction
- Automated testing

---

# Project Overview

Near-Earth Objects (NEOs) are small bodies in the Solar System whose
orbits bring them relatively close to Earth's orbit.

NASA continuously monitors these objects and provides information
about their physical characteristics and close approaches.

This project uses NASA Near-Earth Object data to explore whether
asteroid characteristics can be used to classify objects according
to the provided NASA potentially hazardous asteroid classification.

The project was developed as a portfolio project to demonstrate
practical skills in:

- Python programming
- Data analysis
- Scientific computing
- Machine learning
- Data visualization
- Software testing
- Git/GitHub project organization

> **Important:** This project is an educational machine-learning
> project. It is not an official NASA planetary-defense system and
> does not determine the actual probability of an asteroid impacting
> Earth.

---

# Project Objective

The main objective is to build a reproducible machine-learning
pipeline that:

1. Downloads Near-Earth Object data from NASA.
2. Processes and cleans the raw data.
3. Creates useful machine-learning features.
4. Performs exploratory data analysis.
5. Selects relevant model features.
6. Trains classification models.
7. Compares model performance.
8. Uses cross-validation for additional evaluation.
9. Selects a preliminary final model.
10. Saves the trained model.
11. Uses the saved model for prediction.
12. Validates the project using automated tests.

---

# Project Workflow

```text
                    NASA NeoWs API
                          |
                          v
                   Raw NEO Data
                          |
                          v
                    Data Loading
                          |
                          v
                    Data Cleaning
                          |
                          v
                 Feature Engineering
                          |
                          v
             Exploratory Data Analysis
                          |
                          v
                  Visualization
                          |
                          v
                 Feature Selection
                          |
                          v
                 Train/Test Split
                          |
                          v
                Machine Learning
                   /          \
                  /            \
                 v              v
        Logistic Regression   Random Forest
                 \              /
                  \            /
                   v          v
                 Model Comparison
                          |
                          v
                  Cross-Validation
                          |
                          v
                Final Model Selection
                          |
                          v
                   Save Model
                          |
                          v
                     Prediction
                          |
                          v
                 Automated Testing