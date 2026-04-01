#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Валидация датасета для ЛР3 (туристическое агентство).
Проверяет наличие намеренно внесённых ошибок и формирует карту ошибок.
"""

import pandas as pd
import numpy as np

DATASET_PATH = "Kniga6.csv"
ERRORS_MAP_OUT = "tourism_errors_map_generated.csv"

DUP_PAIRS = [(298, 299), (265, 300), (175, 301), (39, 302)]
LOGIC_IDS = [3, 13, 27, 147]
AGE_OUT_IDS = [200, 246, 277]
CITY_MISSING_IDS = [291]
EDU_MISSING_IDS = [85, 138, 154, 168, 177, 191, 198, 228, 240, 242]
CITY_PREFIX_IDS = [36, 53, 112, 195]

def main():
    df = pd.read_csv(DATASET_PATH)

    errors = []

    # duplicates (ignoring id)
    for a, b in DUP_PAIRS:
        ra = df.loc[df["id"] == a].drop(columns=["id"]).iloc[0]
        rb = df.loc[df["id"] == b].drop(columns=["id"]).iloc[0]
        if not (ra == rb).all():
            print(f"[WARN] Пара {a}/{b} не является дубликатом по полям (кроме id).")
        errors.append({"id": a, "feature":"(row)", "error_type":"duplicate", "description": f"Дубликат (кроме id) с id={b}"})
        errors.append({"id": b, "feature":"(row)", "error_type":"duplicate", "description": f"Дубликат (кроме id) с id={a}"})

    # logic: children_existence <= family_members
    for i in LOGIC_IDS:
        row = df.loc[df["id"] == i].iloc[0]
        if not (row["children_existence"] > row["family_members"] or (row["family_members"] == 1 and row["children_existence"] == 1)):
            print(f"[WARN] id={i}: логическая ошибка не обнаружена по простой проверке.")
        errors.append({"id": i, "feature":"children_existence/family_members", "error_type":"logic",
                       "description":"Нарушение логической связи между количеством детей и членов семьи"})

    # age out of range (примерно: > 100)
    for i in AGE_OUT_IDS:
        age = df.loc[df["id"] == i, "age"].iloc[0]
        if not (age > 100):
            print(f"[WARN] id={i}: age={age} не выглядит как 'вне диапазона'.")
        errors.append({"id": i, "feature":"age", "error_type":"range", "description":"Возраст вне допустимого диапазона"})

    # missing city
    for i in CITY_MISSING_IDS:
        val = df.loc[df["id"] == i, "city"].iloc[0]
        if not (pd.isna(val) or str(val).strip() == ""):
            print(f"[WARN] id={i}: city не пустой.")
        errors.append({"id": i, "feature":"city", "error_type":"missing", "description":"Пустое значение города"})

    # missing education
    for i in EDU_MISSING_IDS:
        val = df.loc[df["id"] == i, "educaction"].iloc[0]
        if not (pd.isna(val) or str(val).strip() == ""):
            print(f"[WARN] id={i}: educaction не пустой.")
        errors.append({"id": i, "feature":"educaction", "error_type":"missing", "description":"Пустое значение уровня образования"})

    # city prefix typo
    for i in CITY_PREFIX_IDS:
        val = str(df.loc[df["id"] == i, "city"].iloc[0])
        if not val.startswith("г."):
            print(f"[WARN] id={i}: city='{val}' не начинается с 'г.'.")
        errors.append({"id": i, "feature":"city", "error_type":"typo", "description":"Орфографическая ошибка: префикс «г.»"})

    out = pd.DataFrame(errors).sort_values(["error_type","id"]).reset_index(drop=True)
    out.to_csv(ERRORS_MAP_OUT, index=False, encoding="utf-8-sig")

    total_errors = len(out)
    print(f"Записей: {len(df)}")
    print(f"Ошибок (по карте): {total_errors}")
    print(f"Доля ошибок: {total_errors/len(df)*100:.2f}%")
    print(f"Сохранено: {ERRORS_MAP_OUT}")

if __name__ == "__main__":
    main()
