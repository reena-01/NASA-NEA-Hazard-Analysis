from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "model_summary.txt"
)


def create_report():
    """Create the final machine-learning model summary report."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = """
============================================================
NASA NEAR-EARTH ASTEROID HAZARD ANALYSIS
MODEL SUMMARY REPORT
============================================================

PROJECT OVERVIEW
------------------------------------------------------------
This project analyzes Near-Earth Object (NEO) data and
explores machine-learning approaches for predicting the
NASA potentially hazardous asteroid classification.

The project demonstrates an end-to-end machine-learning
workflow using Python, Pandas, Scikit-learn, and Matplotlib.


DATASET
------------------------------------------------------------
Source:
NASA Near-Earth Object Web Service (NeoWs)

Dataset size:
36 observations

Model features:
1. absolute_magnitude_h
2. average_diameter_km
3. diameter_range_km
4. relative_velocity_km_s
5. miss_distance_au

Target:
hazard_target


MODELS EVALUATED
------------------------------------------------------------
1. Logistic Regression
2. Random Forest Classifier


TEST SET PERFORMANCE
------------------------------------------------------------

Logistic Regression
Accuracy : 0.8750
Precision: 1.0000
Recall   : 0.5000
F1 Score : 0.6667


Random Forest
Accuracy : 0.8750
Precision: 1.0000
Recall   : 0.5000
F1 Score : 0.6667


CROSS-VALIDATION PERFORMANCE
------------------------------------------------------------

Logistic Regression
Accuracy : 0.7786
Precision: 0.1667
Recall   : 0.3000
F1 Score : 0.2133


Random Forest
Accuracy : 0.8607
Precision: 0.6000
Recall   : 0.7000
F1 Score : 0.6000


FINAL MODEL SELECTION
------------------------------------------------------------
Selected model:
Random Forest

Selection basis:
The Random Forest model demonstrated stronger
5-fold cross-validation performance, particularly
for recall and F1 score.

The two models produced identical results on the
single held-out test set.


MODEL INTERPRETATION
------------------------------------------------------------
The Random Forest model provides a machine-learning
prediction based on asteroid characteristics in the
project dataset.

Feature importance analysis is used to identify which
input variables contribute most to the model's
predictions.

Feature importance represents model-based predictive
importance and should not be interpreted as proof
of physical causation.


LIMITATIONS
------------------------------------------------------------
1. The dataset contains only 36 observations.

2. The dataset represents a limited NASA NeoWs sample.

3. The model learns the provided NASA potentially
   hazardous classification rather than independently
   determining physical asteroid danger.

4. The small dataset means model metrics can vary
   substantially between different samples.

5. The model should not be used for operational
   planetary-defense decisions.

6. The project is intended for educational,
   analytical, and portfolio demonstration purposes.


PROJECT WORKFLOW
------------------------------------------------------------

NASA Data Acquisition
        |
        v
Data Processing
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
Feature Selection
        |
        v
Train / Test Split
        |
        v
Model Training
        |
        v
Model Comparison
        |
        v
Cross-Validation
        |
        v
Final Model Selection
        |
        v
Model Saving
        |
        v
Prediction
        |
        v
Automated Testing


CONCLUSION
------------------------------------------------------------
The project successfully demonstrates an end-to-end
machine-learning workflow for Near-Earth Object analysis.

Random Forest was selected as the preliminary final model
because it demonstrated stronger cross-validation
performance than Logistic Regression.

The result should be considered exploratory because the
current dataset contains only 36 observations.

The project demonstrates practical skills in:

- Python
- Pandas
- NumPy
- Data cleaning
- Feature engineering
- Exploratory data analysis
- Data visualization
- Scikit-learn
- Classification
- Model evaluation
- Cross-validation
- Model persistence
- Automated testing


FINAL STATUS
------------------------------------------------------------
Selected model:
Random Forest

Test set accuracy:
87.50%

5-fold cross-validation accuracy:
86.07%

5-fold cross-validation recall:
70.00%

5-fold cross-validation F1 score:
60.00%

Automated tests:
5 passed

Project status:
Exploratory machine-learning portfolio project.


============================================================
END OF REPORT
============================================================
"""

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            report.strip()
        )

    print("=" * 60)
    print("MODEL SUMMARY REPORT")
    print("=" * 60)

    print("\nReport created successfully!")

    print("\nExact file location:")
    print(REPORT_FILE.resolve())

    print(
        f"\nFile exists: {REPORT_FILE.exists()}"
    )

    print(
        f"File size: {REPORT_FILE.stat().st_size} bytes"
    )

    print("\n" + "=" * 60)


def main():
    """Run the report-generation process."""

    create_report()


if __name__ == "__main__":
    main()