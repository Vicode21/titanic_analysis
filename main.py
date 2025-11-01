# main.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Impostazioni ---
DATA_FILE = "titanic_clean.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --- Caricamento dati ---
df = pd.read_csv(DATA_FILE)

# --- Stampe rapide ---
print("Prime righe del dataset:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nDescrittive (numeriche):")
print(df.describe())

# --- 1) Percentuale sopravvivenza per sesso ---
surv_by_sex = df.groupby("Sex")["Survived"].mean()
print("\nPercentuale di sopravvivenza per sesso:")
print(surv_by_sex)

# Grafico (salvato)
surv_by_sex.plot(kind="bar", title="Percentuale sopravvivenza per sesso")
plt.ylabel("Percentuale (0-1)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "survival_by_sex.png"))
plt.clf()

# --- 2) Percentuale sopravvivenza per classe e sesso (stacked) ---
sop_class = pd.crosstab(
    index=[df["Pclass"], df["Sex"]],
    columns=df["Survived"],
    normalize="index"
)
sop_class.plot(kind="bar", stacked=True, color=["red","green"], figsize=(8,6))
plt.title("Percentuale di sopravvivenza per classe e sesso")
plt.xlabel("(Classe, Sesso)")
plt.ylabel("Percentuale")
plt.legend(title="Survived")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "survival_by_class_sex.png"))
plt.clf()

# --- 3) Distribuzione età e sopravvivenza per fascia ---
bins = [0,10,20,30,40,50,60,70,80]
labels = ["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
age_surv = df.groupby("AgeGroup")["Survived"].mean()
age_surv.plot(kind="bar", title="Percentuale di sopravvivenza per fascia d'età", edgecolor="black")
plt.ylabel("Percentuale")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "survival_by_agegroup.png"))
plt.clf()

print("\nGrafici salvati in:", FIG_DIR)