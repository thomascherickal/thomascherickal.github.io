import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fetch the Titanic dataset online
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Perform data preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df.dropna(inplace=True)

# Convert categorical data to numerical using LabelEncoder
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Scale numerical features using StandardScaler
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the preprocessed dataset to a new file
df.to_csv('preprocessed_titanic.csv', index=False)

print("Preprocessing completed. Preprocessed dataset saved as 'preprocessed_titanic.csv'")