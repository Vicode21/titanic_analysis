import pandas as pd
import matplotlib.pyplot as plt  # <- import corretto

# Carica dataset pulito
df = pd.read_csv("titanic_clean.csv")

# Creiamo le fasce d'età
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

age_surv_sex_class = df.groupby(["AgeGroup", "Pclass", "Sex"])["Survived"].mean()

age_surv_sex_class.unstack().plot(kind="bar", figsize=(10,6))
plt.title("Percentuale sopravvivenza per classe, sesso e età")
plt.xlabel("Fascia età")
plt.ylabel("classe e sesso sopravvissuti")
plt.show()