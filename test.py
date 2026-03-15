import os
import re
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd


try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass

INPUT_CSV = os.path.join(WORK_DIR, INPUT_FILENAME)
OUTPUT_EXCEL = os.path.join(WORK_DIR, f"{TARGET_SYMBOL}.xlsx")


def clean_numeric(val: Any) -> float:
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r"[^\d.-]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_filled_avg(val: Any) -> Tuple[float, float]:
    if pd.isna(val) or "@" not in str(val):
        return 0.0, 0.0
    parts = str(val).split("@")
    qty = clean_numeric(parts[0])
    price = clean_numeric(parts[1])
    return qty, price


def process_data() -> Optional[pd.DataFrame]:
    print(f"当前目标云盘目录: {WORK_DIR}")

    if not os.path.exists(INPUT_CSV):
        print(f"\n❌ 找不到文件: {INPUT_CSV}")
        return None

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="gbk")

    cols_to_fill = ["代码", "方向", "成交时间"]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan).ffill()

    df = df[df["代码"].astype(str).str.startswith(TARGET_SYMBOL)].copy()

    if df.empty:
        print(f"没有找到关于 {TARGET_SYMBOL} 的任何数据。")
        return None

    processed_rows = []
    pending_fee = 0.0

    for _, row in df.iterrows():
        symbol = str(row.get("代码", "")).strip()
        filled_avg_col = row.get("已成交@均价", "")
        fee_val = clean_numeric(row.get("合计费用", 0))

        is_spread_summary = "/" in symbol

        actual_qty = clean_numeric(row.get("成交数量", 0))
        actual_price = clean_numeric(row.get("成交价格", 0))

        is_summary_row = actual_price == 0 and "@" in str(filled_avg_col)

        if is_summary_row:
            pending_fee += fee_val
            continue

        if actual_qty > 0 and actual_price > 0 and not pd.isna(row.get("成交时间")):
            new_row = {
                "Date": pd.to_datetime(re.sub(r"\s*\(.*\)", "", str(row["成交时间"]))),
                "Type": str(row["方向"]).strip(),
                "Symbol": symbol,
                "Quantity": actual_qty,
                "Price": actual_price,
                "Fee": -(fee_val + pending_fee),
            }
            processed_rows.append(new_row)
            pending_fee = 0.0
        else:
            pending_fee += fee_val

    if not processed_rows:
        print(f"没有找到关于 {TARGET_SYMBOL} 的有效成交记录。")
        return None

    final_df = pd.DataFrame(processed_rows)

    final_df["Asset_Type"] = np.where(
        final_df["Symbol"] == TARGET_SYMBOL, "STOCK", "OPTION"
    )
    multiplier = np.where(final_df["Asset_Type"] == "STOCK", 1, 100)

    final_df["Amount"] = np.where(
        final_df["Type"] == "买入",
        -final_df["Quantity"] * final_df["Price"] * multiplier,
        final_df["Quantity"] * final_df["Price"] * multiplier,
    )

    final_df["Date_Str"] = final_df["Date"].dt.strftime("%Y-%m-%d")
    final_df["Year"] = final_df["Date"].dt.strftime("%Y")

    return final_df


def export_to_excel_locally(processed_df: pd.DataFrame) -> None:
    if processed_df is None or processed_df.empty:
        return

    print(f"正在生成 Excel 文件: {OUTPUT_EXCEL}")
    cols = ["Date_Str", "Type", "Symbol", "Quantity", "Price", "Amount", "Fee"]
    headers = ["Date", "Type", "Symbol", "Quantity", "Price", "Amount", "Fee"]

    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            stock = processed_df[processed_df["Asset_Type"] == "STOCK"]
            if not stock.empty:
                out = stock[cols].copy()
                out.columns = headers
                out.to_excel(writer, sheet_name="STOCK", index=False)
                print(" - 已成功写入 STOCK 表")

            option = processed_df[processed_df["Asset_Type"] == "OPTION"]
            if not option.empty:
                for year in sorted(option["Year"].unique(), reverse=True):
                    sheet_name = f"OPTION_{year}"
                    out = option[option["Year"] == year][cols].copy()
                    out.columns = headers
                    out.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f" - 已成功写入 {sheet_name} 表")

        print(f"\n✅ 处理完成！Excel 文件已保存在 Google Drive: {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"❌ 写入 Excel 文件时发生错误: {e}")


if __name__ == "__main__":
    data = process_data()
    if data is not None:
        export_to_excel_locally(data)
