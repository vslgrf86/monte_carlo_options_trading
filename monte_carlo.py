import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# target Param
target_price = 2000000  # Target final price (in euros)
months = 120  # Number of months
years = 10
simulations = 10000  # Number of simulations

# Set the minimum and maximum monthly returns
min_return = -0.16
max_return = 0.08

# Calculate stddev based on the min and max returns
sigma = (max_return - min_return) / (2 * np.sqrt(3))

# initial Param
initial_price = 15000  # Initial investment price (in euros)
mu_lower = 0.00 # bounds
mu_upper = 0.20 # bounds

# Define a tolerance level for finding mu
tolerance = 100  # Tolerance level for the final price

# Create arrays to store the simulated prices, monthly returns, and reinvested profits
prices = np.zeros((simulations, months))
monthly_returns = np.zeros((simulations, months - 1))
reinvested_profits = np.zeros((simulations, months))

prices[:, 0] = initial_price

# Perform the binary search to find the optimal mu
while mu_upper - mu_lower > 0.0001:
    mu = (mu_lower + mu_upper) / 2  # Binary search for mu
    for i in range(1, months):
        random_returns = np.random.uniform(min_return, max_return, simulations)
        prices[:, i] = prices[:, i - 1] * (1 + mu + random_returns)
        monthly_returns[:, i - 1] = mu + random_returns

    for j in range(simulations):
        # Calculate monthly profit/loss
        monthly_return = (prices[j, -1] - prices[j, -2]) / prices[j, -2]

        # Reinvest profits, but hold losses until they reverse
        if monthly_return > 0:
            reinvested_profits[j, :] += prices[j, -1] * monthly_return

    average_price = np.mean(prices[:, -1])

    if average_price < target_price - tolerance:
        mu_lower = mu
    elif average_price > target_price + tolerance:
        mu_upper = mu
    else:
        break

# Statistics of the simulated final prices
average_price = np.mean(prices[:, -1])
median_price = np.median(prices[:, -1])
max_price = np.max(prices[:, -1])
min_price = np.min(prices[:, -1])
percentile_10 = np.percentile(prices[:, -1], 10)
percentile_25 = np.percentile(prices[:, -1], 25)

# Calculate the average monthly return needed to reach the target price
average_monthly_return = (target_price / initial_price) ** (1 / months) - 1

# Calculate the average annual return needed to reach the target price
average_annual_return = (target_price / initial_price) ** (1 / years) - 1

# Calculate the sum of unrealized losses after 120 months
unrealized_losses = np.sum(np.where(prices[:, -1] < initial_price, initial_price - prices[:, -1], 0))

# DF to store the final vslues of the model. Save  to .xlsx
results_df = pd.DataFrame({
    "Simulation": range(1, simulations + 1),
    "Final Price (Millions EUR)": prices[:, -1] / 1e6,  # Display in millions of euros
    "Initial Price (EUR)": initial_price / 1e6  # Display in millions of euros
})
results_df.to_excel("MC_final_prices.xlsx", index=False)

# DF to store the monthly returns of the model. Save  to .xlsx
returns_df = pd.DataFrame(monthly_returns, columns=[f"Month {i + 1}" for i in range(months - 1)])
returns_df.to_excel("MC_monthly_prices.xlsx", index=False)

# Plot
plt.figure(figsize=(20, 10))
for i in range(simulations):
    plt.plot(prices[i, :] / 1e6)  # Divisio to display in millions of euros
plt.axhline(average_price / 1e6, color='r', linestyle='--', label='Average')
plt.axhline(median_price / 1e6, color='g', linestyle='--', label='Median')
plt.axhline(max_price / 1e6, color='b', linestyle='--', label='Max')
plt.axhline(min_price / 1e6, color='y', linestyle='--', label='Min')
plt.axhline(percentile_10 / 1e6, color='m', linestyle='--', label='10th Percentile')
plt.axhline(percentile_25 / 1e6, color='c', linestyle='--', label='25th Percentile')
plt.xlabel("Months")
plt.ylabel("Future Value (Millions EUR)")  
plt.title("Simulated Value Paths with Reinvestments, Compounding, and Holding of Losses until Reversal (10,000 Simulations)")
plt.legend()
plt.grid(True)
plt.show()
