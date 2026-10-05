from sklearn.preprocessing import StandardScaler

X = [
    [25, 50000],
    [31, 55000],
    [42, 70000],
    [28, 60000]
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)