# Ensemble Model with Random Forest Regressor for US Home Price Analysis

## Introduction

In this documentation, we will walk you through the process of building an ensemble model using a Random Forest Regressor with hyperparameter tuning to analyze and predict key factors that influence US home prices on a national scale over the last 20 years. This data science project aims to explain how these factors have impacted home prices.

## Problem Statement

Understanding the factors influencing US home prices is vital for various stakeholders, including homeowners, real estate investors, and policymakers. By developing a data science model, we can gain insights into the relationships between these factors and home prices, enabling better decision-making in the real estate market.

## Implementation Steps

Let's break down the implementation into several steps:

### 1. Data Acquisition

We start by obtaining the dataset named `CSUSHPISA.csv`. This dataset contains historical home price data along with associated dates.

```python
# Load the provided dataset
data = pd.read_csv('CSUSHPISA.csv')
```
### 2. Data Preprocessing
