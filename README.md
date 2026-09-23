# 🚗 Motor Vehicle Insurance — Exploratory Data Analysis

### A Beginner's Guide to Understanding Insurance Data

This project performs an **Exploratory Data Analysis (EDA)** of motor vehicle insurance policy data to understand customer characteristics, vehicle attributes, insurance premiums, and claims.

The objective is to explore the dataset, identify patterns and relationships, detect unusual observations, and generate insights that could support future insurance analytics and modeling projects.

---

## 📌 Project Overview

Insurance companies collect large amounts of information about:

* Customers
* Vehicles
* Insurance policies
* Premiums
* Claims
* Vehicle characteristics

Before building predictive models or making business decisions, it is important to understand the underlying data.

This project uses **Exploratory Data Analysis (EDA)** to investigate the structure and behavior of motor vehicle insurance data.

The analysis focuses on questions such as:

> How are insurance premiums distributed?

> How frequently do policies generate claims?

> Does vehicle age relate to premium?

> Which fuel types are most common?

> Do more powerful vehicles have higher premiums?

---

# 🔎 What is Exploratory Data Analysis?

**Exploratory Data Analysis (EDA)** is the process of investigating a dataset before applying advanced statistical or machine learning techniques.

EDA helps us:

* ✅ Understand the structure and contents of the data
* ✅ Identify missing values
* ✅ Detect outliers and anomalies
* ✅ Understand variable distributions
* ✅ Find patterns and relationships
* ✅ Check assumptions
* ✅ Generate insights for further analysis

A typical EDA workflow looks like:

```text
Raw Dataset
     │
     ▼
Understand Structure
     │
     ▼
Check Data Quality
     │
     ▼
Explore Individual Variables
     │
     ▼
Analyze Relationships
     │
     ▼
Identify Patterns & Outliers
     │
     ▼
Generate Business Insights
```

---

# 📊 Dataset

The project uses a **Motor Vehicle Insurance dataset** available through the **Mendeley Data Repository**.

### Dataset Overview

| Attribute    | Description                  |
| ------------ | ---------------------------- |
| Dataset      | Motor Vehicle Insurance Data |
| Source       | Mendeley Data Repository     |
| Approx. Rows | ~105,000 policy records      |
| Variables    | 30 columns                   |
| Time Period  | 2013–2019                    |
| Domain       | Motor Vehicle Insurance      |

### 🔗 Dataset Source

**[Motor Vehicle Insurance Data — Mendeley Data](https://data.mendeley.com/)**

> The exact dataset download/version used for this analysis should be referenced in the project files or notebook if multiple versions are available.

---

# 🚘 Key Variables

The dataset contains information covering insurance policies, customers, vehicles, premiums, and claims.

Some important variables include:

| Variable              | Description                            |
| --------------------- | -------------------------------------- |
| Premium (€)           | Insurance premium paid for the policy  |
| Claims Cost           | Cost associated with insurance claims  |
| Claims Count          | Number of claims                       |
| Vehicle Age           | Age of the insured vehicle             |
| Vehicle Power         | Power of the vehicle                   |
| Fuel Type             | Fuel category of the vehicle           |
| Customer Demographics | Customer-related characteristics       |
| Policy Details        | Information about the insurance policy |

---

# 🎯 Project Objectives

The primary objective is to understand the insurance dataset through systematic EDA.

The analysis aims to:

1. Understand the dataset structure
2. Assess data quality
3. Identify missing values
4. Examine numerical variables
5. Explore categorical variables
6. Analyze premium distributions
7. Investigate claims behavior
8. Examine relationships between premium and claims
9. Explore vehicle characteristics
10. Identify potential correlations
11. Generate meaningful observations
12. Identify areas for future analysis

---

# 📝 EDA Process

The notebook follows a **15-step EDA workflow**.

## 1. Import Libraries

The analysis begins by importing the required Python libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 2. Load the Data

The insurance dataset is loaded into a Pandas DataFrame.

```python
df = pd.read_csv("motor_vehicle_insurance.csv")
```

---

## 3. Check Dataset Structure

Initial inspection helps understand:

* Number of rows
* Number of columns
* Column names
* Data types
* Sample records

Typical commands include:

```python
df.head()
df.shape
df.info()
df.dtypes
```

---

## 4. Find Missing Values

Missing-value analysis is performed to understand data completeness.

```python
df.isnull().sum()
```

This helps identify variables that may require:

* Imputation
* Removal
* Transformation
* Further investigation

---

## 5. Summary Statistics

Numerical variables are examined using descriptive statistics.

```python
df.describe()
```

This provides information such as:

* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

---

# 💶 6. Premium Distribution

The distribution of insurance premiums is explored using histograms and other visualizations.

Questions include:

* What is the typical premium?
* Is the distribution symmetric?
* Are there unusually high premiums?
* Are premiums heavily skewed?

Example visualization:

```python
sns.histplot(df["Premium"], kde=True)
plt.title("Distribution of Insurance Premiums")
plt.show()
```

---

# 🧾 7. Claims Analysis

The project explores insurance claim frequency and claim costs.

Questions include:

* How many policies have claims?
* How many policies have multiple claims?
* What is the distribution of claim costs?
* Are claims concentrated among a small number of policies?

This provides an initial understanding of the risk characteristics represented in the dataset.

---

# 💰 8. Premium vs Claims

The relationship between insurance premiums and claims is explored.

For example:

```python
sns.scatterplot(
    data=df,
    x="Claims Cost",
    y="Premium"
)

plt.title("Premium vs Claims Cost")
plt.show()
```

This analysis helps identify whether higher claim costs tend to occur alongside higher premiums.

> A relationship observed in EDA should not automatically be interpreted as causation.

---

# 🚗 9. Vehicle Age

Vehicle age is analyzed to understand the characteristics of vehicles in the dataset.

The analysis examines:

* Distribution of vehicle ages
* Typical vehicle age
* Very old or very new vehicles
* Potential relationships between vehicle age and premium

---

# ⛽ 10. Fuel Types

Fuel type is explored as a categorical variable.

The analysis identifies:

* Most common fuel types
* Number of policies by fuel type
* Premium differences across fuel categories
* Potential differences in claims behavior

Example:

```python
sns.countplot(data=df, x="Fuel Type")
plt.title("Vehicle Fuel Type Distribution")
plt.xticks(rotation=45)
plt.show()
```

---

# 👥 11. Premium by Vehicle Age

Premium levels are compared across different vehicle-age groups.

The objective is to explore whether insurance premiums vary depending on vehicle age.

Possible visualization:

```python
sns.boxplot(
    data=df,
    x="Vehicle Age",
    y="Premium"
)

plt.title("Premium by Vehicle Age")
plt.show()
```

For datasets with many unique vehicle-age values, age groups may be created to make the analysis easier to interpret.

---

# 🧾 12. Claims by Vehicle Age

The project also examines whether claim frequency or claim costs vary across vehicle-age groups.

This can help identify patterns that may be useful for future risk analysis.

Examples of questions:

* Do newer vehicles have different claim patterns?
* Are older vehicles associated with higher claim costs?
* Are differences consistent across the dataset?

---

# ⚙️ 13. Vehicle Power

Vehicle power is analyzed to understand whether more powerful vehicles are associated with different premium levels or claims behavior.

Questions include:

* What is the distribution of vehicle power?
* Are high-powered vehicles common?
* Does premium increase with vehicle power?
* Are powerful vehicles associated with different claim patterns?

---

# 🔗 14. Correlation Analysis

Correlation analysis is used to examine relationships between numerical variables.

For example:

```python
correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()
```

Correlation can help identify variables that move together.

However:

> **Correlation does not imply causation.**

A strong correlation does not prove that one variable directly causes another.

---

# 💡 15. Key Findings

The final section of the notebook summarizes the major observations from the analysis.

Examples of findings to investigate include:

* Distribution and skewness of insurance premiums
* Percentage of policies with claims
* Claim-cost distribution
* Most common vehicle fuel types
* Vehicle-age patterns
* Relationship between vehicle power and premium
* Potential correlations between policy characteristics
* Possible outliers or unusual observations

The final findings should be based on the actual results produced by the notebook rather than assumptions about the dataset.

---

# ❓ Key Questions

The EDA is designed around several practical insurance questions.

### Question 1

**How are insurance premiums distributed?**

We examine the central tendency, spread, skewness, and potential outliers of premium values.

### Question 2

**How many policies have claims?**

Claim frequency is analyzed to understand the proportion of policies associated with claims.

### Question 3

**Does vehicle age affect premium?**

Premiums are compared across vehicle-age groups to identify potential relationships.

### Question 4

**Which fuel type is most common?**

Fuel-type frequencies are visualized and compared.

### Question 5

**Do powerful cars cost more to insure?**

Vehicle power is compared with premium values to investigate potential relationships.

---

# 📈 Visualizations

The EDA notebook includes visualizations covering:

* Premium distribution
* Claims distribution
* Claim frequency
* Premium vs claims
* Vehicle age distribution
* Fuel-type distribution
* Premium by vehicle age
* Claims by vehicle age
* Vehicle power distribution
* Vehicle power vs premium
* Correlation matrix
* Additional categorical and numerical analysis

These visualizations help turn raw insurance records into understandable patterns.

---

# 🛠️ Tools & Libraries

## 🐼 Pandas

Used for:

* Data loading
* Data cleaning
* Data transformation
* Aggregation
* Statistical analysis

---

## 📊 Matplotlib

Used for:

* Histograms
* Bar charts
* Scatter plots
* General data visualization

---

## 📈 Seaborn

Used for:

* Statistical visualizations
* Distribution plots
* Box plots
* Count plots
* Correlation heatmaps

---

## 🐍 Python

Python is used as the primary programming language for the complete analysis.

---

# 🗂️ Suggested Project Structure

```text
Motor-Vehicle-Insurance-EDA/
│
├── data/
│   └── motor_vehicle_insurance.csv
│
├── notebooks/
│   └── motor_vehicle_insurance_eda.ipynb
│
├── images/
│   └── charts/
│
├── requirements.txt
│
└── README.md
```

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Motor-Vehicle-Insurance-EDA
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/motor_vehicle_insurance_eda.ipynb
```

Run the notebook cells from top to bottom.

---

# 🔮 What's Next?

EDA is the first step in a broader insurance analytics workflow.

Once the dataset has been explored, several advanced projects can be developed.

---

## 1️⃣ Predictive Modeling

Build machine-learning models to predict:

* Whether a policy will generate a claim
* Expected claim cost
* Potential insurance risk

Possible algorithms include:

* Logistic Regression
* Decision Trees
* Random Forest
* Gradient Boosting

---

## 2️⃣ Customer Segmentation

Customers can be grouped based on characteristics such as:

* Vehicle age
* Vehicle power
* Premium
* Claim history
* Demographics
* Policy characteristics

This can reveal different customer or risk profiles.

---

## 3️⃣ Pricing Optimization

EDA findings can support future analysis of insurance pricing.

Potential objectives include:

* Understanding premium drivers
* Identifying risk factors
* Improving pricing models
* Evaluating premium adequacy

---

## 4️⃣ Fraud Detection

Claims data can be analyzed to identify unusual patterns.

Potential areas include:

* Unusually high claim costs
* Abnormal claim frequency
* Suspicious combinations of policy characteristics
* Outlier behavior

---

# ⚠️ Important Considerations

This project is an **exploratory analysis**, not a production insurance pricing or underwriting model.

The relationships identified during EDA should be investigated further before being used for business decisions.

In particular:

* Correlation does not imply causation.
* Outliers may represent legitimate policies rather than errors.
* Historical patterns may not continue in future periods.
* Additional variables may be required for reliable risk modeling.
* Insurance decisions should consider appropriate regulatory and actuarial requirements.

---

# 🎓 Learning Outcomes

By completing this project, you will gain practical experience with:

* Loading real-world datasets
* Understanding data structures
* Data quality assessment
* Missing-value analysis
* Descriptive statistics
* Univariate analysis
* Bivariate analysis
* Categorical analysis
* Distribution analysis
* Outlier identification
* Correlation analysis
* Data visualization
* Business-oriented insight generation

---

# 📌 Project Summary

This project demonstrates how **Exploratory Data Analysis can transform a large collection of insurance policy records into understandable business insights**.

The workflow starts with raw motor vehicle insurance data and progresses through:

```text
Raw Insurance Data
        ↓
Data Understanding
        ↓
Data Quality Checks
        ↓
Descriptive Statistics
        ↓
Univariate Analysis
        ↓
Bivariate Analysis
        ↓
Correlation Analysis
        ↓
Visualizations
        ↓
Key Findings
        ↓
Future Modeling Opportunities
```

The analysis provides a foundation for future work in **insurance risk modeling, customer segmentation, pricing analytics, and fraud detection**.

---

# 📚 Data Source

**Mendeley Data Repository**

Motor Vehicle Insurance Dataset

[Visit Mendeley Data](https://data.mendeley.com/)

---

## 👤 Author

**Sahil Sharma**

Motor Vehicle Insurance — Exploratory Data Analysis

---

⭐ If you found this project useful, consider giving the repository a star!
