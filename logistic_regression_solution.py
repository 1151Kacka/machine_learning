"""
LOGISTICKÁ REGRESE – Projde student zkouškou? (0 = ne, 1 = ano)
Logistická regrese nevrací číslo, ale PRAVDĚPODOBNOST příslušnosti ke třídě (0 nebo 1).
"""
 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
df = pd.read_csv("dataset.csv")
 
# Nahradíme textové chyby hodnotou NaN
df.replace(["?", "invalid", "NaN"], pd.NA, inplace=True)
 
# Převedeme na čísla
numeric_cols = ["study_hours", "sleep_hours", "attendance", "previous_score",
                "coffee_cups", "final_score"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
 
# Nesmyslné hodnoty docházky
df.loc[df["attendance"] > 100, "attendance"] = pd.NA
df.loc[df["attendance"] < 0,   "attendance"] = pd.NA
 
# Duplicity
df.drop_duplicates(inplace=True)
 
# Chybějící hodnoty → průměr
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())
 
# ÚLOHA 3 – Vytvoření klasifikační proměnné
 
# Pravidlo školy: 75+ bodů = projde
# True/False -> 1/0 pomocí .astype(int)
df["passed"] = (df["final_score"] >= 75).astype(int)
 
print(" ROZDĚLENÍ: prošli vs neprošli ")
print(df["passed"].value_counts())
# Proč kategorii místo čísla? -> Chceme rozhodnutí (projde/neprojde), ne přesné skóre.
 
# ÚLOHA 4 – Explorace: rozdíly mezi skupinami
 
print("\n PRŮMĚRY PODLE SKUPINY (prošli=1 / neprošli=0) ")
print(df.groupby("passed")[["study_hours", "sleep_hours", "attendance"]].mean().round(2))
# Očekáváme: studenti co prošli -> více hodin studia, lepší docházka
 
# ÚLOHA 5 – Trénink modelu logistické regrese
 
print("\n LOGISTICKÁ REGRESE ")
 
# Vstupy
X = df[["study_hours", "sleep_hours", "attendance"]]
# Výstup: 0 nebo 1
y = df["passed"]
 
model = LogisticRegression()
model.fit(X, y)    # model se naučí, které kombinace vstupů vedou k 0 nebo 1
 
predictions = model.predict(X)   # predikuje 0/1 pro každého studenta
 
# ÚLOHA 6 – Vyhodnocení modelu
 
acc = accuracy_score(y, predictions)
print(f"Přesnost (accuracy): {acc:.0%}")   # % správně klasifikovaných
print("\nDetailní report:")
print(classification_report(y, predictions, target_names=["Neprošel (0)", "Prošel (1)"]))
# precision = z těch co model označil jako '1', kolik jich opravdu prošlo
# recall    = ze všech co prošli, kolik model správně chytil
 
# ÚLOHA 7 – Pravděpodobnosti
 
print("\n PRAVDĚPODOBNOSTI PRŮCHODU ")
proba = model.predict_proba(X)   # vrátí [P(neprojde), P(projde)] pro každého studenta
df["prob_pass"] = proba[:, 1]    # vezmeme druhou hodnotu = pravděpodobnost průchodu
 
print(df[["student_id", "final_score", "passed", "prob_pass"]].round(2))
# Pravděpodobnost 0.8 znamená: model si je z 80 % jistý, že student projde.
# Threshold (hranice): standardně 0.5 → nad ní = projde. Můžeme ji změnit.
 
# ÚLOHA 8 – Koeficienty modelu (interpretace)
 
print("\n KOEFICIENTY LOGISTICKÉ REGRESE ")
for feature, coef in zip(X.columns, model.coef_[0]):
    smer = "zvyšuje" if coef > 0 else "snižuje"
    print(f"  {feature}: {coef:.3f}  → {smer} šanci na průchod")
# Kladný koeficient = více tohoto vstup → větší šance na průchod (a naopak)
 
# ÚLOHA 9 – Rozhodování: systém varování
 
print("\n VAROVNÝ SYSTÉM ")
# Navrhujeme: pokud pravděpodobnost průchodu < 0.6 -> varovat studenta
THRESHOLD = 0.6
df["warning"] = df["prob_pass"] < THRESHOLD
rizikovi = df[df["warning"]][["student_id", "study_hours", "attendance", "prob_pass"]]
print(f"Studenti v riziku (P < {THRESHOLD}):")
print(rizikovi)
# V praxi: škola by těmto studentům nabídla konzultace nebo doučování
 
# ÚLOHA 10 – Vlastní analýza
 
print("\n VLASTNÍ ANALÝZA: Vliv kávy na průchod ")
# Průměrný počet šálků kávy podle toho, zda student prošel
coffee_analysis = df.groupby("passed")["coffee_cups"].agg(["mean", "max"]).round(2)
coffee_analysis.index = ["Neprošel", "Prošel"]
print(coffee_analysis)
# Pokud neprošlí pijí víc kávy -> možná příznak stresu nebo špatného spánku
 
