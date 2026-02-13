
import ast
import pandas as pd
from typing import Any, Optional

from recipe_search.io.models import RecipeRecord


class DatasetLoader:
    def __init__(
        self,
        max_rows: Optional[int] = None,
        random_sample: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.max_rows = max_rows
        self.random_sample = random_sample
        self.seed = seed

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        if self.max_rows is not None:
            df = df.head(self.max_rows).copy()

        if self.random_sample is not None and self.random_sample < len(df):
            df = df.sample(
                n=self.random_sample,
                random_state=self.seed).copy()

        for col in ("ingredients", "directions", "NER"):
            if col in df.columns:
                df[col] = df[col].apply(self._parse_listlike)

        if "title" in df.columns:
            df["title"] = df["title"].fillna("").astype(str)

        return df.reset_index(drop=True)

    def to_records(self, df: pd.DataFrame) -> list[RecipeRecord]:
        records: list[RecipeRecord] = []

        df = df.reset_index(drop=True)

        for recipe_id, row in enumerate(df.itertuples(index=False)):
            title = str(getattr(row, "title", "")).strip()

            ingredients = self._parse_listlike(getattr(row, "ingredients", None))
            ner = self._parse_listlike(getattr(row, "NER", None))

            link = self._safe_str(getattr(row, "link", None))

            records.append(
                RecipeRecord(
                    recipe_id=recipe_id,
                    title=title,
                    ingredients=[str(x).strip() for x in ingredients if str(x).strip()],
                    ner=[str(x).strip() for x in ner if str(x).strip()],
                    # source=source,
                    link=link,
                )
            )

        return records


    def _parse_listlike(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, float) and pd.isna(value):
            return []

        if isinstance(value, list):
            return [str(x) for x in value]

        s = str(value).strip()

        if not s:
            return []

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

        if "," in s:
            return [
                p.strip().strip("'\"")
                for p in s.split(",")
                if p.strip()]

        return [s]

    def _safe_str(self, v: Any) -> Optional[str]:
        if v is None:
            return None

        if isinstance(v, float) and pd.isna(v):
            return None

        s = str(v).strip()

        return s if s else None


if __name__ == "__main__":
    PATH = "data/recipenlg.csv"

    loader = DatasetLoader(max_rows=5)

    df = loader.load_csv(PATH)
    print("OK: CSV loaded")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(3))

    for col in ("ingredients", "NER", "directions"):
        if col in df.columns:
            print(f"\n{col} type:", type(df.loc[0, col]))
            print(f"{col} value:", df.loc[0, col])

    records = loader.to_records(df)
    print("\nOK: Records created:", len(records))
    print("First record:", records[0])