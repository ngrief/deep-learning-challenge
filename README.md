# deep-learning-challenge
# Alphabet Soup Charity Funding Predictor

## Overview of the Analysis
The purpose of this analysis is to develop a **binary classification model** that can predict whether applicants for Alphabet Soup funding will be successful. Using machine learning and deep learning techniques, we analyze historical data to identify patterns that distinguish **successful** funding recipients from **unsuccessful** ones.

---

## Results

### **Data Preprocessing**
#### **What variable(s) are the target(s) for your model?**
- The target variable is **`IS_SUCCESSFUL`**, which indicates whether the organization successfully utilized the provided funding.

#### **What variable(s) are the features for your model?**
- All other columns, except identification fields, serve as features. Key features include:
  - **APPLICATION_TYPE** (Type of application submitted)
  - **AFFILIATION** (Affiliated sector of industry)
  - **CLASSIFICATION** (Government classification)
  - **USE_CASE** (Intended use of funds)
  - **ORGANIZATION** (Type of organization)
  - **STATUS** (Active status)
  - **INCOME_AMT** (Income classification)
  - **SPECIAL_CONSIDERATIONS** (Special conditions)
  - **ASK_AMT** (Amount of funding requested)

#### **What variable(s) should be removed from the input data because they are neither targets nor features?**
- The following columns were removed:
  - **EIN** (Employer Identification Number) – Unique to each entity and does not contribute to prediction.
  - **NAME** (Organization name) – Does not provide meaningful numerical input for classification.

---

### **Compiling, Training, and Evaluating the Model**
#### **How many neurons, layers, and activation functions did you select for your neural network model, and why?**
- **Model Architecture:**
  - **Input Layer**: Matches the number of input features.
  - **Hidden Layers**:
    - **Layer 1**: 256 neurons, `ReLU` activation
    - **Layer 2**: 128 neurons, `ReLU` activation
    - **Layer 3**: 64 neurons, `ReLU` activation
    - **Layer 4**: 32 neurons, `ReLU` activation (added for better feature extraction)
  - **Output Layer**: 1 neuron, `sigmoid` activation (since it’s a binary classification problem).

- **Why this architecture?**
  - **ReLU activation** is chosen for hidden layers because it helps mitigate the vanishing gradient problem.
  - **Sigmoid activation** is used in the output layer for binary classification.
  - **Four hidden layers** provide a good balance between model complexity and overfitting prevention.

#### **Were you able to achieve the target model performance?**
- The goal was to achieve **75%+ accuracy**.
- The final model achieved **~73.25% accuracy**, which is close to the goal but did not surpass the threshold.

#### **What steps did you take in your attempts to increase model performance?**
- Several optimizations were attempted:
  1. **Increasing Neurons** – Higher capacity to learn complex patterns.
  2. **Adding Dropout Layers** – To reduce overfitting.
  3. **Trying Different Activation Functions** – Switched between `ReLU` and `LeakyReLU`.
  4. **Adjusting Learning Rate** – Lowered learning rate (`0.0005`) for smoother weight updates.
  5. **Increasing Training Epochs** – Extended training but kept **early stopping** enabled.
  6. **Tuning Batch Size** – Experimented with `32` vs `64` batch sizes.
  7. **Binning Rare Categories** – Combined low-frequency categorical values into an `"Other"` group.

---

## **Summary**
- The deep learning model **performed well but did not exceed 75% accuracy**.
- **Challenges:** Imbalanced dataset, complexity of categorical variables.
- **Strengths:** The model successfully captured key patterns in the dataset and showed stable learning.

### **Alternative Model Recommendation**
A deep learning model is **not always the best** for structured tabular data. A more suitable approach could be:
- **Random Forest Classifier** – Handles categorical data well and reduces overfitting.
- **XGBoost (Extreme Gradient Boosting)** – Often outperforms deep learning on tabular data.
- **Logistic Regression** – A simpler model that can still provide high accuracy if feature selection is done correctly.

**Why?**
- **Tree-based models (XGBoost, Random Forest) are often better suited for structured categorical data.**
- **They require less tuning and are easier to interpret.**

---

## **Final Thoughts**
- While the **deep learning model** performed well, **a traditional machine learning approach** (such as XGBoost) may be more effective for this dataset.
- Further tuning, additional feature engineering, and **ensemble methods** could improve prediction accuracy.

📌 **Next Steps:**
- Experiment with **XGBoost or Random Forest** to compare performance.
- Perform **feature selection** to eliminate less relevant variables.
- Use **oversampling techniques** (e.g., SMOTE) if data imbalance is affecting results.

---
### **Project Files**
- `AlphabetSoupCharity.ipynb` → Initial model training
- `AlphabetSoupCharity_Optimization.ipynb` → Optimized model
- `AlphabetSoupCharity_Optimization.h5` → Final trained model
- `README.md` → Project documentation
