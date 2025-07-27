import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example: DataFrame with a skewed feature
df = pd.DataFrame({
    'income': [1000, 2000, 3000, 40000, 50000, 600000]
},)

df.hist()
plt.show()
# Apply log-transform
df['income_log'] = np.log1p(df['income'])  # log(1 + x)
# df.hist()
# plt.show()