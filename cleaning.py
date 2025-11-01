import pandas as pd

# --- CARICAMENTO DATI ORIGINALI ---
df = pd.read_csv("titanic_data.csv")

# --- PULIZIA DATI ---
# Rimuoviamo la colonna Cabin (troppi valori mancanti)
df = df.drop(columns=["Cabin"])

# Riempire i valori mancanti di Age e Fare con la mediana
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Controllo valori mancanti
print("\nValori mancanti per colonna:")
print(df.isna().sum())

# Controllo valori unici di Sex e Survived
print("\nValori unici in 'Sex':", df['Sex'].unique())
print("Valori unici in 'Survived':", df['Survived'].unique())

# --- SALVATAGGIO DATASET PULITO ---
df.to_csv("titanic_clean.csv", index=False)
print("\n✅ Dataset pulito salvato come 'titanic_clean.csv'")