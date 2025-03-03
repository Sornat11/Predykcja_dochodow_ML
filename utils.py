from scipy.stats import chi2_contingency, pointbiserialr, spearmanr, pearsonr
import numpy as np
import pandas as pd

def cramers_v(tab):
    chi2, p, dof, expected = chi2_contingency(tab)
    n = tab.sum().sum()
    k, r = tab.shape
    cramers_v = np.sqrt(chi2 / (n * (min(k - 1, r - 1))))
    
    return cramers_v

def describe_variable(data, variable_type, variable_name):
    stats = {
        "index": variable_name,
        "mean": "nie dotyczy",
        "median": "nie dotyczy",
        "mode": "nie dotyczy",
        "std": "nie dotyczy",
        "min": "nie dotyczy",
        "max": "nie dotyczy",
        "unique": "nie dotyczy",
        "missing": data.isna().sum()
    }

    if variable_type == "quantitive":
        stats.update({
            "mean": data.mean(),
            "median": data.median(),
            "mode": data.mode().tolist(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "unique": data.nunique()
        })

    elif variable_type == "ordinal":
        stats.update({
            "median": data.median(),
            "mode": data.mode().tolist(),
            "min": data.min(),
            "max": data.max(),
            "unique": data.nunique()
        })

    elif variable_type == "categorical":
        stats.update({
            "mode": data.mode().tolist(),
            "unique": data.nunique()
        })
    return stats

def cramer_v_two_cols(data1, data2):
    # Tworzenie tabeli kontyngencji
    contingency_table = pd.crosstab(data1, data2)
    
    # Obliczanie testu chi-kwadrat
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Cramér's V
    n = contingency_table.sum().sum()  # Całkowita liczba obserwacji
    v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
    return v

def point_biserial(data1, data2):
    # Zakłada, że data1 to zmienna binarna, a data2 to zmienna ilościowa
    corr, _ = pointbiserialr(data1, data2)
    return corr

def spearman(data1, data2):
    # Obliczanie korelacji rang Spearmana
    corr, _ = spearmanr(data1, data2)
    return corr

def pearson(data1, data2):
    # Obliczanie korelacji Pearsona
    corr, _ = pearsonr(data1, data2)
    return corr

def dict_to_corr_matrix(corr_dict):
    # Znajdź wszystkie unikalne nazwy kolumn
    columns = sorted(set(key[0] for key in corr_dict.keys()).union(key[1] for key in corr_dict.keys()))
    
    # Stwórz pustą macierz DataFrame z indeksami i kolumnami
    corr_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    
    # Wypełnij macierz wartościami z słownika
    for (row, col), value in corr_dict.items():
        corr_matrix.loc[row, col] = value
        corr_matrix.loc[col, row] = value  # Symetria macierzy korelacji
    
    # Uzupełnij przekątną wartościami 1 (korelacja własna)
    np.fill_diagonal(corr_matrix.values, 1)
    
    return corr_matrix

def undersample_data(df, target_column):
    """
    Funkcja wykonuje undersampling dla zmiennej docelowej, aby zbilansować liczność klas.

    :param df: DataFrame zawierający dane.
    :param target_column: Nazwa kolumny zmiennej docelowej.
    :return: Zbilansowany DataFrame.
    """
    # Podział danych na klasy
    class_0 = df[df[target_column] == 0]
    class_1 = df[df[target_column] == 1]

    # Ustalenie minimalnej liczby rekordów w klasach
    min_size = min(len(class_0), len(class_1))

    # Losowy undersampling
    class_0_undersampled = class_0.sample(n=min_size, random_state=42)
    class_1_undersampled = class_1.sample(n=min_size, random_state=42)

    # Połączenie zbilansowanych klas
    balanced_df = pd.concat([class_0_undersampled, class_1_undersampled]).sample(frac=1, random_state=42)

    return balanced_df

def apply_one_hot_encoding(df, columns):
    for col in columns:
        encoded_df = pd.get_dummies(df[col], dtype=int, dummy_na = False)
        encoded_df.loc[df[col].isnull(), :] = np.nan
        df = df.drop(col, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
    return df