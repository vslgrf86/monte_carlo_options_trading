import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set the random seed for NumPy for reproducibility
np.random.seed(42)

# Target Parameters
target_price = 2000000  # Target final price (in euros)
months = 120  # Number of months
years = 10  # Number of years
simulations = 10000  # Number of simulations

# Set the minimum and maximum monthly returns
min_return = -0.16
max_return = 0.08

# Calculate standard deviation (sigma) based on the min and max returns
sigma = (max_return - min_return) / (2 * np.sqrt(3))

# Initial Parameters
initial_price = 15000  # Initial investment price (in euros)
mu_lower = 0.00  # Lower bound for mu
mu_upper = 0.20  # Upper bound for mu

# Define a tolerance level for finding mu
tolerance = 100  # Tolerance level for the final price

# Create arrays to store the simulated prices, monthly returns, and reinvested profits
prices = np.zeros((simulations, months))
monthly_returns = np.zeros((simulations, months))
reinvested_profits = np.zeros((simulations, months))

prices[:, 0] = initial_price

# Perform binary search to find the optimal mu
while mu_upper - mu_lower > 0.0001:
    mu = (mu_lower + mu_upper) / 2  # Binary search for mu
    for i in range(1, months):
        random_returns = np.random.uniform(min_return, max_return, simulations)
        prices[:, i] = prices[:, i - 1] * (1 + mu + random_returns)
        monthly_returns[:, i - 1] = mu + random_returns

    # Calculate monthly profit/loss and reinvest profits
    monthly_returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
    reinvested_profits = np.where(monthly_returns > 0, prices[:, 1:] * monthly_returns, 0)

    # Calculate annual returns
    annual_returns = np.zeros((simulations, years))
    for j in range(simulations):
        for year in range(years):
            annual_returns[j, year] = (prices[j, (year + 1) * 12 - 1] - prices[j, year * 12]) / prices[j, year * 12]

    average_price = np.mean(prices[:, -1])

    if average_price < target_price - tolerance:
        mu_lower = mu
    elif average_price > target_price + tolerance:
        mu_upper = mu
    else:
        break

# Calculate statistics of the simulated final prices
average_price = np.mean(prices[:, -1])
median_price = np.median(prices[:, -1])
max_price = np.max(prices[:, -1])
min_price = np.min(prices[:, -1])
percentile_10 = np.percentile(prices[:, -1], 10)
percentile_25 = np.percentile(prices[:, -1], 25)
percentile_75 = np.percentile(prices[:, -1], 75)

# Calculate the average monthly return needed to reach the target price
average_monthly_return = (target_price / initial_price) ** (1 / months) - 1

# Calculate the average annual return needed to reach the target price
average_annual_return = (target_price / initial_price) ** (1 / years) - 1

# Calculate the sum of unrealized losses after 120 months
unrealized_losses = np.sum(np.where(prices[:, -1] < initial_price, initial_price - prices[:, -1], 0))

# Create DataFrames to store the final values of the model
final_returns_df = pd.DataFrame({
    "Simulation": range(1, simulations + 1),
    "Final Return (Millions EUR)": prices[:, -1] / 1e6,  # Display in millions of euros
    "Initial Value (EUR)": initial_price  # Display in euros
})

annual_returns_df = pd.DataFrame(annual_returns, columns=[f'Year {year + 1}' for year in range(years)])

monthly_returns_df = pd.DataFrame(monthly_returns, columns=[f'Month {i + 1}' for i in range(1, months)])

# Save results to an Excel file
with pd.ExcelWriter("MC_returns.xlsx") as writer:
    final_returns_df.to_excel(writer, sheet_name="Final Returns", index=False)
    annual_returns_df.to_excel(writer, sheet_name="Annual Returns", index=False)
    monthly_returns_df.to_excel(writer, sheet_name="Monthly Returns", index=False)

# Plot
plt.figure(figsize=(20, 10))
for i in range(simulations):
    plt.plot(prices[i, :] / 1e6)  # Division to display in millions of euros

# Add text with statistics to the plot
plt.axhline(average_price / 1e6, color='r', linestyle='--', label='Average')
plt.axhline(median_price / 1e6, color='g', linestyle='--', label='Median')
plt.axhline(max_price / 1e6, color='b', linestyle='--', label='Max')
plt.axhline(min_price / 1e6, color='y', linestyle='--', label='Min')
plt.axhline(percentile_10 / 1e6, color='m', linestyle='--', label='10th Percentile')
plt.axhline(percentile_25 / 1e6, color='c', linestyle='--', label='25th Percentile')
plt.axhline(percentile_75 / 1e6, color='b', linestyle='--', label='75th Percentile')

# Add statistics as text to the plot
plt.text(2, average_price / 1e6, f'Average: {average_price / 1e6:.2f} Million EUR', fontsize=6)
plt.text(2, median_price / 1e6, f'Median: {median_price / 1e6:.2f} Million EUR', fontsize=6)
plt.text(2, max_price / 1e6, f'Max: {max_price / 1e6:.2f} Million EUR', fontsize=6)
plt.text(2, min_price / 1e6, f'Min: {min_price / 1e6:.2f} Million EUR', fontsize=6)
plt.text(2, percentile_10 / 1e6, f'10th Percentile: {percentile_10 / 1e6:.2f} Million EUR', fontsize=6)
plt.text(2, percentile_25 / 1e6, f'25th Percentile: {percentile_25 / 1e6:.2f} Million EUR', fontsize=6)
plt.text(2, percentile_75 / 1e6, f'75th Percentile: {percentile_75 / 1e6:.2f} Million EUR', fontsize=6)

plt.xlabel("Months")
plt.ylabel("Future Value (Millions EUR)")
plt.title("Simulated Value Paths with Reinvestments, Compounding, and Holding of Losses until Reversal (10,000 Simulations)")
plt.legend()
plt.grid(True)

# Print statistics to the console
print("Simulation Statistics:")
print(f"Average Price: {average_price / 1e6:.2f} Million EUR")
print(f"Median Price: {median_price / 1e6:.2f} Million EUR")
print(f"Max Price: {max_price / 1e6:.2f} Million EUR")
print(f"Min Price: {min_price / 1e6:.2f} Million EUR")
print(f"10th Percentile Price: {percentile_10 / 1e6:.2f} Million EUR")
print(f"25th Percentile Price: {percentile_25 / 1e6:.2f} Million EUR")
print(f"75th Percentile Price: {percentile_75 / 1e6:.2f} Million EUR")
print(f"IQR Price: {(percentile_75 - percentile_25) / 1e6:.2f} Million EUR")
print("\nInvestment Statistics:")
print(f"Average Monthly Return Needed: {average_monthly_return:.4f}")
print(f"Average Annual Return Needed: {average_annual_return:.4f}")
print(f"Sum of Unrealized Losses: {unrealized_losses:.2f} EUR")

plt.show()
