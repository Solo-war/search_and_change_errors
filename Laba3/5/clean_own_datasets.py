from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd


# Храню пути проекта и папку для результатов.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "output"


# Задаю имена колонок для файлов без заголовков.
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

MOVIELENS_ITEM_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "genre_unknown",
    "genre_action",
    "genre_adventure",
    "genre_animation",
    "genre_childrens",
    "genre_comedy",
    "genre_crime",
    "genre_documentary",
    "genre_drama",
    "genre_fantasy",
    "genre_film_noir",
    "genre_horror",
    "genre_musical",
    "genre_mystery",
    "genre_romance",
    "genre_sci_fi",
    "genre_thriller",
    "genre_war",
    "genre_western",
]


# Храню одну запись в карте ошибок.
@dataclass
class ErrorRecord:
    dataset: str
    source_file: str
    row_number_source: int
    record_id: str
    field: str
    error_type: str
    current_value: str
    fixed_value: str
    expected_rule: str


def value_to_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


# Пишу найденную ошибку и способ исправления в общем формате.
def add_error(
    errors: list[ErrorRecord],
    *,
    dataset: str,
    source_file: str,
    row_number_source: int,
    record_id: object,
    field: str,
    error_type: str,
    current_value: object,
    fixed_value: object,
    expected_rule: str,
) -> None:
    errors.append(
        ErrorRecord(
            dataset=dataset,
            source_file=source_file,
            row_number_source=row_number_source,
            record_id=value_to_text(record_id),
            field=field,
            error_type=error_type,
            current_value=value_to_text(current_value),
            fixed_value=value_to_text(fixed_value),
            expected_rule=expected_rule,
        )
    )


def mode_without_marker(series: pd.Series, invalid_marker: str) -> str:
    valid = series.astype(str).str.strip()
    valid = valid[valid.ne(invalid_marker)]
    return valid.mode().iat[0]


# Ищу дубликаты, собираю их индексы и пишу удаление в отчет.
def collect_duplicate_indices(
    df: pd.DataFrame,
    subset: list[str],
    *,
    dataset: str,
    source_file: str,
    row_offset: int,
    record_id_column: str | None,
    errors: list[ErrorRecord],
    expected_rule: str,
) -> list[int]:
    seen: dict[tuple[object, ...], int] = {}
    duplicates: list[int] = []

    for index, row in df.iterrows():
        key = tuple(row[column] for column in subset)
        if key in seen:
            first_index = seen[key]
            add_error(
                errors,
                dataset=dataset,
                source_file=source_file,
                row_number_source=index + row_offset,
                record_id=row[record_id_column] if record_id_column else "",
                field="ALL_FIELDS",
                error_type="duplicate_record",
                current_value="-",
                fixed_value="row_removed",
                expected_rule=f"{expected_rule}; дубликат строки {first_index + row_offset}",
            )
            duplicates.append(index)
        else:
            seen[key] = index

    return duplicates


def save_outputs(
    dataset_name: str,
    cleaned_df: pd.DataFrame,
    errors: list[ErrorRecord],
    cleaned_filename: str,
    *,
    summary_rows: list[dict[str, object]],
    source_file: str,
    source_rows: int,
) -> None:
    # Сохраняю очищенный датасет и отдельную карту ошибок по нему.
    dataset_output_dir = OUTPUT_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = dataset_output_dir / cleaned_filename
    errors_path = dataset_output_dir / f"{dataset_name}_errors_map.csv"

    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    errors_df = pd.DataFrame(asdict(error) for error in errors)
    if errors_df.empty:
        errors_df = pd.DataFrame(
            columns=[
                "dataset",
                "source_file",
                "row_number_source",
                "record_id",
                "field",
                "error_type",
                "current_value",
                "fixed_value",
                "expected_rule",
            ]
        )
    else:
        errors_df = errors_df.sort_values(
            ["row_number_source", "field", "error_type"]
        ).reset_index(drop=True)
    errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")

    summary_rows.append(
        {
            "dataset": dataset_name,
            "source_file": source_file,
            "source_rows": source_rows,
            "cleaned_rows": len(cleaned_df),
            "errors_found": len(errors_df),
            "cleaned_file": str(cleaned_path.relative_to(PROJECT_DIR)),
            "errors_map_file": str(errors_path.relative_to(PROJECT_DIR)),
        }
    )


def clean_adult(summary_rows: list[dict[str, object]]) -> list[ErrorRecord]:
    # 1. Читаю исходный файл и создаю журнал ошибок.
    dataset_name = "adult"
    source_file = "adult/adult.data"
    source_path = PROJECT_DIR / source_file

    df = pd.read_csv(
        source_path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
    )
    original_df = df.copy()
    errors: list[ErrorRecord] = []

    # 2. Считаю, чем заменяю маркер пропуска '?'.
    fill_values = {
        "workclass": mode_without_marker(df["workclass"], "?"),
        "occupation": mode_without_marker(df["occupation"], "?"),
        "native_country": mode_without_marker(df["native_country"], "?"),
    }

    # 3. Исправляю пропуски и пишу каждое исправление в errors.
    for column, fill_value in fill_values.items():
        mask = df[column].astype(str).str.strip().eq("?")
        for index in df.index[mask]:
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 1,
                record_id="",
                field=column,
                error_type="missing_value",
                current_value=df.at[index, column],
                fixed_value=fill_value,
                expected_rule=f"{column} должно содержать валидную категорию, не '?'",
            )
            df.at[index, column] = fill_value

    # 4. Удаляю полные дубликаты строк.
    duplicate_indices = collect_duplicate_indices(
        df,
        ADULT_COLUMNS,
        dataset=dataset_name,
        source_file=source_file,
        row_offset=1,
        record_id_column=None,
        errors=errors,
        expected_rule="Полная строка должна быть уникальной",
    )
    if duplicate_indices:
        df = df.drop(index=duplicate_indices).reset_index(drop=True)

    # 5. Сохраняю результат и пишу статистику по датасету.
    save_outputs(
        dataset_name,
        df,
        errors,
        "adult_cleaned.csv",
        summary_rows=summary_rows,
        source_file=source_file,
        source_rows=len(original_df),
    )
    return errors


def clean_bank_marketing(summary_rows: list[dict[str, object]]) -> list[ErrorRecord]:
    # 1. Читаю основной датасет Bank Marketing.
    dataset_name = "bank_marketing"
    source_file = "bank-marketing/bank/bank-full.csv"
    source_path = PROJECT_DIR / source_file

    df = pd.read_csv(source_path, sep=";")
    original_df = df.copy()
    errors: list[ErrorRecord] = []

    # 2. Считаю, чем заменяю 'unknown'.
    fill_values = {
        "job": mode_without_marker(df["job"], "unknown"),
        "education": mode_without_marker(df["education"], "unknown"),
    }

    # 3. Исправляю 'unknown' и пишу каждую замену в отчет.
    for column, fill_value in fill_values.items():
        mask = df[column].astype(str).str.strip().eq("unknown")
        for index in df.index[mask]:
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 2,
                record_id="",
                field=column,
                error_type="missing_value",
                current_value=df.at[index, column],
                fixed_value=fill_value,
                expected_rule=f"{column} должно содержать валидную категорию, не 'unknown'",
            )
            df.at[index, column] = fill_value

    # 4. Удаляю полные дубликаты записей.
    duplicate_indices = collect_duplicate_indices(
        df,
        list(original_df.columns),
        dataset=dataset_name,
        source_file=source_file,
        row_offset=2,
        record_id_column=None,
        errors=errors,
        expected_rule="Полная строка должна быть уникальной",
    )
    if duplicate_indices:
        df = df.drop(index=duplicate_indices).reset_index(drop=True)

    # 5. Сохраняю очищенный датасет и карту ошибок.
    save_outputs(
        dataset_name,
        df,
        errors,
        "bank_marketing_cleaned.csv",
        summary_rows=summary_rows,
        source_file=source_file,
        source_rows=len(original_df),
    )
    return errors


def clean_bank_marketing_additional(summary_rows: list[dict[str, object]]) -> list[ErrorRecord]:
    # 1. Читаю расширенный датасет Bank Marketing Additional.
    dataset_name = "bank_marketing_additional"
    source_file = (
        "bank-marketing/bank-additional/bank-additional/bank-additional-full.csv"
    )
    source_path = PROJECT_DIR / source_file

    df = pd.read_csv(source_path, sep=";")
    original_df = df.copy()
    errors: list[ErrorRecord] = []

    # 2. Считаю замены для маркера 'unknown' в нужных полях.
    fill_values = {
        column: mode_without_marker(df[column], "unknown")
        for column in ["job", "marital", "education", "default", "housing", "loan"]
    }

    # 3. Исправляю 'unknown' и пишу каждое изменение в журнале.
    for column, fill_value in fill_values.items():
        mask = df[column].astype(str).str.strip().eq("unknown")
        for index in df.index[mask]:
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 2,
                record_id="",
                field=column,
                error_type="missing_value",
                current_value=df.at[index, column],
                fixed_value=fill_value,
                expected_rule=f"{column} должно содержать валидную категорию, не 'unknown'",
            )
            df.at[index, column] = fill_value

    # 4. После исправлений удаляю полные дубликаты.
    duplicate_indices = collect_duplicate_indices(
        df,
        list(original_df.columns),
        dataset=dataset_name,
        source_file=source_file,
        row_offset=2,
        record_id_column=None,
        errors=errors,
        expected_rule="Полная строка должна быть уникальной",
    )
    if duplicate_indices:
        df = df.drop(index=duplicate_indices).reset_index(drop=True)

    # 5. Сохраняю очищенные данные и ошибки по этому набору.
    save_outputs(
        dataset_name,
        df,
        errors,
        "bank_marketing_additional_cleaned.csv",
        summary_rows=summary_rows,
        source_file=source_file,
        source_rows=len(original_df),
    )
    return errors


def clean_credit_default(summary_rows: list[dict[str, object]]) -> list[ErrorRecord]:
    # 1. Читаю Excel-датасет по кредитным клиентам.
    dataset_name = "default_credit"
    source_file = "default-of-credit-card-clients.xls"
    source_path = PROJECT_DIR / source_file

    df = pd.read_excel(source_path, header=1, engine="xlrd")
    original_df = df.copy()
    errors: list[ErrorRecord] = []

    # 2. Задаю правила нормализации некорректных кодов категорий.
    education_fix_map = {0: 4, 5: 4, 6: 4}
    marriage_fix_map = {0: 3}

    # 3. Исправляю значения EDUCATION вне допустимых категорий.
    for index, value in df["EDUCATION"].items():
        if int(value) in education_fix_map:
            fixed_value = education_fix_map[int(value)]
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 3,
                record_id=df.at[index, "ID"],
                field="EDUCATION",
                error_type="invalid_category",
                current_value=value,
                fixed_value=fixed_value,
                expected_rule="EDUCATION in {1,2,3,4}; коды 0/5/6 приводятся к 4 (others)",
            )
            df.at[index, "EDUCATION"] = fixed_value

    # 4. Исправляю значения MARRIAGE с недопустимым кодом.
    for index, value in df["MARRIAGE"].items():
        if int(value) in marriage_fix_map:
            fixed_value = marriage_fix_map[int(value)]
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 3,
                record_id=df.at[index, "ID"],
                field="MARRIAGE",
                error_type="invalid_category",
                current_value=value,
                fixed_value=fixed_value,
                expected_rule="MARRIAGE in {1,2,3}; код 0 приводится к 3 (others)",
            )
            df.at[index, "MARRIAGE"] = fixed_value

    # 5. Удаляю дубликаты по ID.
    duplicate_indices = collect_duplicate_indices(
        original_df,
        ["ID"],
        dataset=dataset_name,
        source_file=source_file,
        row_offset=3,
        record_id_column="ID",
        errors=errors,
        expected_rule="ID должен быть уникальным",
    )
    if duplicate_indices:
        df = df.drop(index=duplicate_indices).reset_index(drop=True)

    # 6. Перед сохранением возвращаю числовые поля к int.
    integer_columns = df.columns
    df[integer_columns] = df[integer_columns].astype(int)

    # 7. Сохраняю очищенный датасет и журнал ошибок.
    save_outputs(
        dataset_name,
        df,
        errors,
        "default_credit_cleaned.csv",
        summary_rows=summary_rows,
        source_file=source_file,
        source_rows=len(original_df),
    )
    return errors


def make_imdb_url(title: str) -> str:
    return f"http://us.imdb.com/M/title-exact?{quote(title, safe='()')}"


def clean_movielens_items(summary_rows: list[dict[str, object]]) -> list[ErrorRecord]:
    # 1. Читаю список фильмов MovieLens и создаю журнал ошибок.
    dataset_name = "movielens_items"
    source_file = "ml-100k/ml-100k/u.item"
    source_path = PROJECT_DIR / source_file

    df = pd.read_csv(
        source_path,
        sep="|",
        header=None,
        names=MOVIELENS_ITEM_COLUMNS,
        encoding="latin-1",
    )
    original_df = df.copy()
    errors: list[ErrorRecord] = []

    rows_to_drop: list[int] = []

    # 2. Для каждой строки решаю: удаляю битую карточку или восстанавливаю ссылку.
    for index in df.index:
        title = value_to_text(df.at[index, "title"]).strip()
        release_date = value_to_text(df.at[index, "release_date"]).strip()
        imdb_url = value_to_text(df.at[index, "imdb_url"]).strip()

        if title.lower() == "unknown" and not release_date and not imdb_url:
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 1,
                record_id=df.at[index, "movie_id"],
                field="title",
                error_type="placeholder_value",
                current_value=title,
                fixed_value="row_removed",
                expected_rule="title должен содержать реальное название фильма",
            )
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 1,
                record_id=df.at[index, "movie_id"],
                field="release_date",
                error_type="missing_value",
                current_value=release_date,
                fixed_value="row_removed",
                expected_rule="release_date должно быть заполнено для карточки фильма",
            )
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 1,
                record_id=df.at[index, "movie_id"],
                field="imdb_url",
                error_type="missing_value",
                current_value=imdb_url,
                fixed_value="row_removed",
                expected_rule="imdb_url должно быть заполнено для карточки фильма",
            )
            rows_to_drop.append(index)
            continue

        # 3. Если карточка валидная, но imdb_url пустой, достраиваю его из title.
        if not imdb_url:
            fixed_url = make_imdb_url(title)
            add_error(
                errors,
                dataset=dataset_name,
                source_file=source_file,
                row_number_source=index + 1,
                record_id=df.at[index, "movie_id"],
                field="imdb_url",
                error_type="missing_value",
                current_value=imdb_url,
                fixed_value=fixed_url,
                expected_rule="imdb_url должно быть заполнено валидной ссылкой на IMDb",
            )
            df.at[index, "imdb_url"] = fixed_url

    # 4. Удаляю дубликаты по movie_id и объединяю их с уже помеченными строками.
    duplicate_indices = collect_duplicate_indices(
        original_df,
        ["movie_id"],
        dataset=dataset_name,
        source_file=source_file,
        row_offset=1,
        record_id_column="movie_id",
        errors=errors,
        expected_rule="movie_id должен быть уникальным",
    )
    rows_to_drop.extend(duplicate_indices)

    if rows_to_drop:
        df = df.drop(index=sorted(set(rows_to_drop))).reset_index(drop=True)

    # 5. Сохраняю очищенный файл и карту ошибок.
    save_outputs(
        dataset_name,
        df,
        errors,
        "movielens_items_cleaned.csv",
        summary_rows=summary_rows,
        source_file=source_file,
        source_rows=len(original_df),
    )
    return errors


# Храню список доступных обработчиков датасетов.
DATASET_CLEANERS = {
    "adult": clean_adult,
    "bank_marketing": clean_bank_marketing,
    "bank_marketing_additional": clean_bank_marketing_additional,
    "default_credit": clean_credit_default,
    "movielens_items": clean_movielens_items,
}


def parse_args() -> argparse.Namespace:
    # Даю выбрать конкретные датасеты вместо полного запуска.
    parser = argparse.ArgumentParser(
        description="Находит и исправляет ошибки в 5 выбранных датасетах."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_CLEANERS),
        help="Обработать только указанный датасет. Можно передать несколько раз.",
    )
    return parser.parse_args()


def main() -> None:
    # 1. Определяю, какие датасеты нужно обработать.
    args = parse_args()
    selected_datasets = args.dataset or list(DATASET_CLEANERS)

    # 2. Запускаю очистку каждого выбранного набора и коплю общий журнал.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    all_errors: list[ErrorRecord] = []

    for dataset_name in selected_datasets:
        dataset_errors = DATASET_CLEANERS[dataset_name](summary_rows)
        all_errors.extend(dataset_errors)

    # 3. Собираю сводку и объединенную карту ошибок по всем датасетам.
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    all_errors_df = pd.DataFrame(asdict(error) for error in all_errors)
    all_errors_path = OUTPUT_DIR / "all_datasets_errors_map.csv"
    if all_errors_df.empty:
        all_errors_df = pd.DataFrame(
            columns=[
                "dataset",
                "source_file",
                "row_number_source",
                "record_id",
                "field",
                "error_type",
                "current_value",
                "fixed_value",
                "expected_rule",
            ]
        )
    else:
        all_errors_df = all_errors_df.sort_values(
            ["dataset", "row_number_source", "field", "error_type"]
        ).reset_index(drop=True)
    all_errors_df.to_csv(all_errors_path, index=False, encoding="utf-8-sig")

    # 4. Печатаю пути к файлам результатов и краткую статистику.
    print("Готово.")
    print(f"Сводка: {summary_path}")
    print(f"Общая карта ошибок: {all_errors_path}")
    for row in summary_rows:
        print(
            f"{row['dataset']}: исходных строк={row['source_rows']}, "
            f"исправлено ошибок={row['errors_found']}, "
            f"строк после очистки={row['cleaned_rows']}"
        )


if __name__ == "__main__":
    main()
