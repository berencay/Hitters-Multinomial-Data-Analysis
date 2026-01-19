# Hitters Multinomial Data Analysis (R)

**Live report (GitHub Pages):** https://berencay.github.io/Hitters-Multinomial-Data-Analysis/  
**Source code:** https://github.com/berencay/Hitters-Multinomial-Data-Analysis

## Overview
This project builds a **multinomial logistic regression** model to classify baseball players into **four salary-level segments** using performance statistics from the **Hitters** dataset.

Salary levels (created from salary quartiles):
- RookieBudget (bottom 25%)
- SolidStarter (25–50%)
- BeenThereDoneThat (50–75%)
- TooRichToPitch (top 25%)

## Workflow
- Cleaned the dataset by removing rows with missing salary values.
- Created the multiclass target variable (`SalaryLevel`) based on salary quartiles.
- Performed exploratory data analysis (EDA) using boxplots to understand relationships between predictors and salary groups.
- Trained a multinomial logistic regression model using `nnet::multinom`.
- Evaluated model performance using **LOOCV** and **5-Fold Cross-Validation**.
- Improved generalization by applying **backward stepwise selection (AIC)** via `MASS::stepAIC` to build a reduced model.

## Results (Accuracy)
- Full model
  - Training accuracy: **69.88%**
  - LOOCV accuracy: **59.85%**
  - 5-Fold CV accuracy: **~61%**
- Reduced model (AIC backward selection)
  - Training accuracy: **67.95%**
  - LOOCV accuracy: **63.71%**
  - 5-Fold CV accuracy: **61.39%**

## Tech Stack
- R / R Markdown
- tidyverse, ggplot2
- nnet (multinomial logistic regression)
- caret (evaluation utilities)
- MASS (stepAIC for model selection)

## Repository Structure
- `index.html` — rendered report used by GitHub Pages
- `report/ProjectWork_We R Who We R.Rmd` — full reproducible analysis (source)
- `data/Hitters.csv` — dataset used in the analysis
- `docs/GroupProject.pdf` — project write-up (optional)
