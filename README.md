# 🛒 Instacart Shopping Basket Analysis

## 📌 Project Summary
This project aims to discover customer purchasing patterns using Instacart's global shopping data. The model, designed with the **Apriori (FP-Growth)** logic, calculates product associations (Association Rules) and generates concrete marketing modules and shelf arrangement suggestions for the retail world.

## 🎯 Business Objectives & Mission
- **Cross-Selling:** Finding product pairs that trigger each other.
- **Shelf Planning:** Optimizing product placement in in-store or digital distribution.
- **Bulk Offers:** Presenting products with high "Lift" as bundles.

## 📊 Statistical Metrics
Three main performance indicators were used in our analysis:
- **Support:** Issues seen in all baskets of a product document.
- **Confidence:** The probability of purchasing product B, which includes product A.
- **Lift:** How much the sale of product A increases the sales option of product B (Lift > 1 = Positive Relationship).

## 🚀 Technical Pipeline
1. **ETL:** Combining 5 different tables ("orders", "products", "aisles", "departments", "order_products") with the Main Table.
2. **EDA:** Visualizing the most popular products, aisles, and order times.
3. **MBA Modeling:** Removing 50,000 basket ratios from the bundle.
4. **Work Output:** Classifying the analysis results according to marketing methods (Push notifications, Bundle Deal, etc.).

## 💡 Key Findings
Some of the strongest associations revealed by the analysis are:
- **Organic Raspberry → Organic Strawberry** (Increase: 2.89)
- **Organic Fuji Apple → Banana** (Increase: 2.45)
- **Kirby Cucumber → Banana** (Increase: 2.13)
