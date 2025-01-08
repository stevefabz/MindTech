# The MindTech project: Machine Learning and Analytics for Mental Health in Tech Workplaces

This project explores mental health in tech workplaces using machine learning, clustering, and interactive visualizations. The goal is to uncover actionable insights that help create supportive, healthier environments for employees.

---

## Project Overview

Mental health in the tech industry often takes a backseat despite its critical importance. This project tackles this issue through three key analyses:

1. **Predictive Modeling**: Identifying factors that drive individuals to seek mental health treatment.
2. **Clustering**: Grouping workplaces based on their mental health practices.
3. **Power BI Analytics**: Visualizing data to highlight systemic trends and improve accessibility.

The analyses leverage survey data on workplace policies, work interference levels, and treatment-seeking behavior.

---

## Files and Resources

- **Code Scripts**:
  - `predictive_modeling.py`: Random Forest implementation for treatment prediction.
  - `clustering_analysis.py`: KMeans clustering to map workplace mental health practices.
- **Data Visualizations**:
  - `cluster_visualization.png`: Scatter plot of workplace clusters based on PCA.
  - Power BI dashboards in the report: Interactive insights into treatment rates, workplace size, and leave policies.
- **Logs and Outputs**:
  - `model_training.txt`: Logs for predictive modeling.
  - `clustering_log.txt`: Logs for clustering analysis.
  - `cluster_centers.txt`: Characteristics of workplace mental health clusters.

---

## Key Insights

### **1. Predictive Modeling**
- **Objective**: Predict who is likely to seek mental health treatment.
- **Key Findings**:
  - Work interference is the most significant factor.
  - Family history and care options also play pivotal roles.
- **Outcome**: The model achieved an accuracy of **81%**, providing a reliable foundation for targeted interventions.

### **2. Clustering Analysis**
- **Objective**: Map workplaces based on mental health-related attributes.
- **Key Findings**:
  - Identified four clusters ranging from highly supportive workplaces to those needing significant improvements.
  - Cluster 2 stands out with exceptional mental health practices, serving as a benchmark for others.
- **Outcome**: Offers a framework for understanding systemic differences in workplace support.

### **3. Power BI Analytics**
- **Objective**: Create interactive dashboards for exploring systemic trends.
- **Key Findings**:
  - 50.52% of respondents sought mental health treatment.
  - Smaller workplaces have higher treatment-seeking rates, indicating resource gaps.
  - Policies around leave and supervisor support need significant improvement.
- **Outcome**: Empowers decision-makers to act on visually accessible insights.

---

## How to Run

### **Environment Setup**
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure survey data (`survey.csv`) is available in the project directory.

### **Scripts**
1. **Run Predictive Modeling**:
   ```bash
   python predictive_modeling.py
   ```
   Outputs include the classification report, accuracy score, and feature importance.

2. **Run Clustering Analysis**:
   ```bash
   python clustering_analysis.py
   ```
   Outputs include cluster visualizations and cluster centers.

3. **Power BI Dashboards**:
   Open `Mental Health in Tech Workplaces.pbix` in Power BI Desktop to explore interactive dashboards.


### Power BI Insights
- **Treatment Rates**: 50.52% of respondents sought treatment.
- **Workplace Size**: Smaller workplaces need stronger mental health support systems.
- **Leave Policies**: A significant area for improvement, with many employees finding policies difficult to navigate.


---

## Final Thoughts
Mental health in tech workplaces is more than a personal challenge; it is an organizational responsibility. By combining machine learning, clustering, and interactive dashboards, this project provides actionable insights to foster healthier, more inclusive environments. Together, we can ensure that innovation and well-being go hand in hand.

