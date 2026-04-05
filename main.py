import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass

TARGET_SYMBOL = "NVDA"
INPUT_FILENAME = "raw_data.csv"
WORK_DIR = "/content/drive/MyDrive/Test"
INPUT_CSV = os.path.join(WORK_DIR, INPUT_FILENAME)
OUTPUT_EXCEL = os.path.join(WORK_DIR, f"{TARGET_SYMBOL}.xlsx")

# 预编译正则表达式，提升循环内部重复调用时的性能
_NUMERIC_CLEAN_RE = re.compile(r"[^\d.-]")
# 预编译期权前缀匹配正则
_SYMBOL_PREFIX_RE = re.compile(r"^([A-Z]+\d+)[CP]")


def clean_numeric(val: Any) -> float:

    if pd.isna(val) or val == "":
        return 0.0

    if isinstance(val, (int, float)):
        return float(val)

    res = _NUMERIC_CLEAN_RE.sub("", str(val))

    try:
        return float(res)
    except ValueError:
        return 0.0


def analyze_strategy(symbol: str, name: str, direction: str) -> str:
    symbol = symbol.upper()
    name = str(name).strip()
    direction = str(direction).strip()

    if len(symbol) <= 5 or not any(char.isdigit() for char in symbol):
        return "正股交易 (Stock)"

    opt_type = ""
    if "C" in symbol:
        opt_type = "Call"
    elif "P" in symbol:
        opt_type = "Put"

    if "策略" in name:
        if "垂直" in name:
            if opt_type == "Call":
                strategy = (
                    "熊市看涨价差 (Bear Call)"
                    if direction == "卖出"
                    else "牛市看涨价差 (Bull Call)"
                )
            elif opt_type == "Put":
                strategy = (
                    "牛市看跌价差 (Bull Put)"
                    if direction == "卖出"
                    else "熊市看跌价差 (Bear Put)"
                )
            else:
                strategy = "垂直策略 (Vertical)"
        elif "日历" in name:
            strategy = "日历策略 (Calendar)"
        elif "跨式" in name:
            strategy = "跨式策略 (Straddle)"
        else:
            strategy = name.split()[-1] if " " in name else name
        return strategy

    if direction == "买入" and opt_type == "Call":
        strategy = "买入看涨 (Long Call)"
    elif direction == "卖出" and opt_type == "Call":
        strategy = "卖出看涨 (Short Call)"
    elif direction == "买入" and opt_type == "Put":
        strategy = "买入看跌 (Long Put)"
    elif direction == "卖出" and opt_type == "Put":
        strategy = "卖出看跌 (Short Put)"
    else:
        if "/" in symbol:
            strategy = "组合策略 (Combo)"
        else:
            strategy = "单腿期权"

    return strategy


def _prepare_dataframe(df: pd.DataFrame, target_symbol: str) -> Optional[pd.DataFrame]:

    df = df.copy()
    cols_to_fill = [col for col in ["方向", "代码", "名称"] if col in df.columns]

    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].replace("", np.nan).ffill()

    df_filtered = df[df["代码"].astype(str).str.startswith(target_symbol)]

    if df_filtered.empty:
        print(f"没有找到关于 {target_symbol} 的任何数据。")
        return None

    return df_filtered.reset_index(drop=True)


def _extract_trades(df: pd.DataFrame) -> Optional[pd.DataFrame]:

    processed_rows = []
    pending_fee = 0.0

    strategy = "未知策略"
    active_combo_prefix = ""
    active_combo_qty = 0

    # 使用 to_dict('records') 替代 iterrows，速度提升 10x-50x 且完美兼容 row.get()
    for row in df.to_dict("records"):
        order_qty = str(row.get("订单数量", "")).strip()
        fee = clean_numeric(row.get("合计费用", 0))
        direction = str(row.get("方向", "")).strip()
        symbol = str(row.get("代码", "")).strip()
        name = str(row.get("名称", "")).strip()

        prefix_match = _SYMBOL_PREFIX_RE.match(symbol)
        symbol_prefix = prefix_match.group(1) if prefix_match else symbol

        if "组" in order_qty:
            strategy = analyze_strategy(symbol, name, direction)
            active_combo_prefix = symbol_prefix
            active_combo_qty = clean_numeric(order_qty)
            pending_fee += fee
            continue

        elif order_qty != "":
            current_qty = clean_numeric(order_qty)
            if (
                active_combo_prefix
                and symbol_prefix == active_combo_prefix
                and current_qty == active_combo_qty
            ):
                pass
            else:
                strategy = analyze_strategy(symbol, name, direction)
                active_combo_prefix = ""
                active_combo_qty = 0

        actual_qty = clean_numeric(row.get("成交数量", 0))
        actual_price = clean_numeric(row.get("成交价格", 0))
        actual_amount = clean_numeric(row.get("成交金额", 0))

        if actual_qty > 0 and actual_price > 0 and actual_amount > 0:
            processed_rows.append(
                {
                    "Raw_Date": str(row.get("成交时间", "")),
                    "Type": direction,
                    "Symbol": symbol,
                    "Strategy": strategy,
                    "Quantity": actual_qty,
                    "Raw_Quantity_Price": actual_qty * actual_price,
                    "Raw_Amount": actual_amount,
                    "Fee": -(fee + pending_fee),
                }
            )
            pending_fee = 0.0
        else:
            pending_fee += fee

    if not processed_rows:
        return None

    return pd.DataFrame(processed_rows)


def _aggregate_trades(df: pd.DataFrame) -> pd.DataFrame:

    clean_dates = pd.to_datetime(
        df["Raw_Date"].str.replace(r"\s*\(.*\)", "", regex=True)
    )
    df = df.assign(
        Date=clean_dates.dt.strftime("%Y-%m-%d"), Year=clean_dates.dt.year.astype(str)
    )
    cols_to_group = ["Date", "Year", "Type", "Symbol", "Strategy"]
    df_grouped = df.groupby(cols_to_group, as_index=False).agg(
        {
            "Quantity": "sum",
            "Fee": "sum",
            "Raw_Quantity_Price": "sum",
            "Raw_Amount": "sum",
        }
    )
    df_grouped["Price"] = (
        df_grouped["Raw_Quantity_Price"] / df_grouped["Quantity"]
    ).round(4)
    return df_grouped.drop(columns=["Raw_Quantity_Price"])


def _validate_result(df: pd.DataFrame, target_symbol: str) -> pd.DataFrame:

    asset_type = np.where(df["Symbol"] == target_symbol, "STOCK", "OPTION")
    multiplier = np.where(asset_type == "STOCK", 1, 100)
    trade_sign = np.where(df["Type"] == "买入", -1, 1)
    actual_amount = df["Raw_Amount"] * trade_sign
    theoretical_amount = df["Quantity"] * df["Price"] * multiplier * trade_sign
    df = (
        df.assign(
            Asset_Type=asset_type,
            Amount=actual_amount,
            Theoretical_Amount=theoretical_amount.round(2),
            Validation_Diff=abs(theoretical_amount - actual_amount).round(2),
        )
        .drop(columns=["Raw_Amount"])
        .sort_values(by="Date", ascending=False)
        .reset_index(drop=True)
    )
    return df


def process_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:

    prepared_df = _prepare_dataframe(df, TARGET_SYMBOL)

    if prepared_df is None:
        return None

    extracted_df = _extract_trades(prepared_df)

    if extracted_df is None:
        return None

    aggregated_df = _aggregate_trades(extracted_df)
    validated_df = _validate_result(aggregated_df, TARGET_SYMBOL)
    return validated_df


def export_excel(df: pd.DataFrame) -> None:

    if df is None or df.empty:
        return

    print(f"正在生成 Excel 文件: {OUTPUT_EXCEL}")
    cols = ["Date", "Type", "Symbol", "Strategy", "Quantity", "Price", "Amount", "Fee"]
    log_col_map = {
        "Date": "Date",
        "Type": "Type",
        "Symbol": "Symbol",
        "Quantity": "Quantity",
        "Price": "Avg_Price",
        "Amount": "Actual_Amount",
        "Theoretical_Amount": "Expected_Amount",
        "Validation_Diff": "Diff_Warning",
    }

    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            stock_df = df[df["Asset_Type"] == "STOCK"]

            if not stock_df.empty:
                stock_df[cols].to_excel(writer, sheet_name="STOCK", index=False)
                print(" ✅ 已成功写入 STOCK 表")

            option_df = df[df["Asset_Type"] == "OPTION"]

            if not option_df.empty:
                for year in sorted(option_df["Year"].unique(), reverse=True):
                    sheet_name = f"OPTION_{year}"
                    option_year_df = option_df[option_df["Year"] == year]
                    option_year_df[cols].to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )
                    print(f" ✅ 已成功写入 {sheet_name} 表")

            validation_df = df[df["Validation_Diff"] > 0.5]

            if not validation_df.empty:
                log_df = validation_df[list(log_col_map.keys())].rename(
                    columns=log_col_map
                )
                log_df.to_excel(writer, sheet_name="VALIDATION", index=False)
                print(
                    f" ⚠️ 警告：发现 {len(validation_df)} 笔资金对账异常！已生成 'LOG' 标签页。"
                )
            else:
                print(" ✅ 数据验证通过：所有理论计算金额与券商实际扣款匹配！")

        print(f"✅ 处理完成！Excel 文件已保存在 Google Drive: {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"❌ 写入 Excel 文件时发生错误: {e}")


def main():

    print(f"当前目标云盘目录: {WORK_DIR}")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到文件: {INPUT_CSV}")
        return

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="gbk")

    data = process_data(df)

    if data is not None:
        export_excel(data)


if __name__ == "__main__":
    main()
