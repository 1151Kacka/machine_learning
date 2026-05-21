"""
LINEÁRNÍ REGRESE = Předpověď výsledku studenta, hledá přímku, která nejlépe vystihuje vztah mezi vstupy a výstupem.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ÚLOHA 1 

df = pd.read_csv("dataset.csv")

print("1")
print(df.head())          # prvních 5 řádků
print(df.info())          # typy sloupců, počet ne-null hodnot
print(f"\nŘádků: {df.shape[0]}, Sloupců: {df.shape[1]}")

# ÚLOHA 2 – Čištění dat

print("\n ČIŠTĚNÍ DAT ")

# Nahradíme textové "chyby" hodnotou NaN, aby pandas věděl, že jde o chybějící data
df.replace(["?", "invalid", "NaN"], pd.NA, inplace=True)

# Převedeme sloupce na čísla (coerce = co nejde převést → NaN)
numeric_cols = ["study_hours", "sleep_hours", "attendance", "previous_score",
                "coffee_cups", "final_score"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Nesmyslné hodnoty: attendance mimo rozsah 0–100
df.loc[df["attendance"] > 100, "attendance"] = pd.NA
df.loc[df["attendance"] < 0,   "attendance"] = pd.NA

# Odstraníme duplicity (stejný student dvakrát)
df.drop_duplicates(inplace=True)

# Chybějící hodnoty nahradíme průměrem daného sloupce
# (medián by byl robustnější, ale průměr je jednodušší na pochopení)
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

print("Data po čištění:")
print(df)

# ÚLOHA 3 – Vztahy: study_hours / sleep_hours vs final_score

print("\n VZTAHY")
print("Korelace study_hours × final_score:", round(df["study_hours"].corr(df["final_score"]), 2))
print("Korelace sleep_hours × final_score:", round(df["sleep_hours"].corr(df["final_score"]), 2))
# Vyšší číslo = silnější vztah. Kladné = čím víc, tím lepší výsledek.

# ÚLOHA 4 – Vliv docházky
print("\nVLIV DOCHÁZKY")
# Rozdělíme studenty na nízkou (<80%) a vysokou (>=80%) docházku
df["attendance_group"] = df["attendance"].apply(lambda x: "vysoka" if x >= 80 else "nizka")
print(df.groupby("attendance_group")["final_score"].mean())
# Očekáváme, že vyšší docházka -> lepší skóre

# ÚLOHA 5 – Vliv kávy

print("\nVLIV KÁVY")
print("Korelace coffee_cups × final_score:", round(df["coffee_cups"].corr(df["final_score"]), 2))
# Záporná korelace by znamenala: víc kávy -> horší výsledky (stres, nespavost)

# ÚLOHA 6 – Celková korelační matice

print("\nKORELACE matice")
print(df[numeric_cols].corr().round(2))

# ÚLOHA 7 – Model: Lineární regrese

print("\n LINEÁRNÍ REGRESE ")

# Vstupy (X) = co model dostane jako informace
X = df[["study_hours", "sleep_hours", "attendance"]]
# Výstup (y) = co chceme předpovědět
y = df["final_score"]

model = LinearRegression()
model.fit(X, y)        # trénink: model najde nejlepší přímku

predictions = model.predict(X)   # predikce na stejných datech

# Vyhodnocení modelu
mae = mean_absolute_error(y, predictions)   # průměrná chyba v bodech
r2  = r2_score(y, predictions)              # jak dobře model vysvětluje data (0–1)

print(f"Průměrná chyba (MAE): {mae:.2f} bodů")
print(f"R² skóre: {r2:.2f}  (1.0 = perfektní, 0 = náhodné)")

# ÚLOHA 8 – Koeficienty modelu (interpretace)

print("\n=KOEFICIENTY MODELU ")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.3f}")
print(f"  (intercept/konstanta): {model.intercept_:.3f}")
# Koeficient říká: o kolik bodů se změní final_score, když vstup vzroste o 1 jednotku

# ÚLOHA 9 – Vlastní analýza: 

print("\n VLASTNÍ ANALÝZA: Studenti nad průměrem v učení I spánku")
avg_study = df["study_hours"].mean()
avg_sleep = df["sleep_hours"].mean()

# Podmínka: studuje víc než průměr A spí víc než průměr
superstars = df[(df["study_hours"] > avg_study) & (df["sleep_hours"] > avg_sleep)]
print(f"Počet takových studentů: {len(superstars)}")
print(f"Jejich průměrné skóre:   {superstars['final_score'].mean():.1f}")
print(f"Průměr zbytku:           {df[~df.index.isin(superstars.index)]['final_score'].mean():.1f}")