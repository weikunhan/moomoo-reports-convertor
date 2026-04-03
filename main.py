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


def clean_numeric(val: Any) -> float:
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    res = re.sub(r"[^\d.-]", "", str(val))
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


# ================= 数据处理拆分模块 =================


def _prepare_dataframe(df: pd.DataFrame, target_symbol: str) -> Optional[pd.DataFrame]:
    """步骤 1: 填充缺失数据并过滤出目标标的"""
    cols_to_fill = ["代码", "方向", "名称"]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan).ffill()

    df_filtered = df[df["代码"].astype(str).str.startswith(target_symbol)].copy()

    if df_filtered.empty:
        print(f"没有找到关于 {target_symbol} 的任何数据。")
        return None
    return df_filtered


def _extract_raw_trades(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """步骤 2: 逐行提取交易明细，只做原始字符串提取"""
    processed_rows = []
    pending_fee = 0.0

    current_strategy = "未知策略"
    active_combo_prefix = ""
    active_combo_qty = 0

    for _, row in df.iterrows():
        order_qty = str(row.get("订单数量", "")).strip()
        fee = clean_numeric(row.get("合计费用", 0))
        direction = str(row.get("方向", "")).strip()
        symbol = str(row.get("代码", "")).strip()
        name = str(row.get("名称", "")).strip()

        prefix_match = re.match(r"^([A-Z]+\d+)[CP]", symbol)
        symbol_prefix = prefix_match.group(1) if prefix_match else symbol

        if "组" in order_qty:
            current_strategy = analyze_strategy(symbol, name, direction)
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
                current_strategy = analyze_strategy(symbol, name, direction)
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
                    "Strategy": current_strategy,
                    "Quantity": actual_qty,
                    "Raw_Price": actual_qty * actual_price,
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
    """步骤 3: 集中处理时间格式并聚合拆分订单"""
    # 1. 提取时间并生成辅助年份列
    df["Date"] = pd.to_datetime(
        df["Raw_Date"].str.replace(r"\s*\(.*\)", "", regex=True)
    )
    df["Year"] = df["Date"].dt.strftime("%Y")

    # 直接原地覆写 Date 列为字符串，彻底抛弃 Date_Str
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 直接将 Date 作为分组主键
    group_cols = ["Date", "Year", "Type", "Symbol", "Strategy"]

    df_grouped = df.groupby(group_cols, as_index=False).agg(
        {"Quantity": "sum", "Fee": "sum", "Raw_Price": "sum", "Raw_Amount": "sum"}
    )

    df_grouped["Price"] = (df_grouped["Raw_Price"] / df_grouped["Quantity"]).round(4)
    df_grouped = df_grouped.drop(columns=["Raw_Price"])

    return df_grouped


def _calculate_final_metrics(df: pd.DataFrame, target_symbol: str) -> pd.DataFrame:
    """步骤 4: 赋予资金正负号、计算资产类别并执行严谨的资金验证"""
    df["Amount"] = np.where(df["Type"] == "买入", -df["Raw_Amount"], df["Raw_Amount"])

    df["Asset_Type"] = np.where(df["Symbol"] == target_symbol, "STOCK", "OPTION")
    multiplier = np.where(df["Asset_Type"] == "STOCK", 1, 100)

    # 核心验证逻辑
    theoretical_amount = np.where(
        df["Type"] == "买入",
        -df["Quantity"] * df["Price"] * multiplier,
        df["Quantity"] * df["Price"] * multiplier,
    )

    df["Validation_Diff"] = abs(theoretical_amount - df["Amount"]).round(2)
    df["Theoretical_Amount"] = theoretical_amount.round(2)

    df = df.drop(columns=["Raw_Amount"])
    df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)

    return df


def process_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """主控流水线: 按顺序调度各个数据处理模块"""
    # 1. 准备数据
    df = _prepare_dataframe(df, TARGET_SYMBOL)
    if df is None:
        return None

    # 2. 提取明细与费用归集
    raw_trades_df = _extract_raw_trades(df)
    if raw_trades_df is None:
        return None

    # 3. 分组聚合处理拆分订单
    aggregated_df = _aggregate_trades(raw_trades_df)

    # 4. 计算验证结果与最终金额
    final_df = _calculate_final_metrics(aggregated_df, TARGET_SYMBOL)

    return final_df


# ===================================================


def export_excel(processed_df: pd.DataFrame) -> None:

    if processed_df is None or processed_df.empty:
        return

    print(f"正在生成 Excel 文件: {OUTPUT_EXCEL}")
    cols = ["Date", "Type", "Symbol", "Strategy", "Quantity", "Price", "Amount", "Fee"]
    val_cols = [
        "Date",
        "Type",
        "Symbol",
        "Quantity",
        "Price",
        "Amount",
        "Theoretical_Amount",
        "Validation_Diff",
    ]
    val_headers = [
        "Date",
        "Type",
        "Symbol",
        "Quantity",
        "Avg_Price",
        "Actual_Amount",
        "Expected_Amount",
        "Diff_Warning",
    ]

    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            stock_df = processed_df[processed_df["Asset_Type"] == "STOCK"]

            if not stock_df.empty:
                res = stock_df[cols].copy()
                res.to_excel(writer, sheet_name="STOCK", index=False)
                print(" ✅ 已成功写入 STOCK 表")

            option_df = processed_df[processed_df["Asset_Type"] == "OPTION"]

            if not option_df.empty:
                for year in sorted(option_df["Year"].unique(), reverse=True):
                    sheet_name = f"OPTION_{year}"
                    res = option_df[option_df["Year"] == year][cols].copy()
                    res.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f" ✅ 已成功写入 {sheet_name} 表")

            validation_df = processed_df[processed_df["Validation_Diff"] > 0.5]

            if not validation_df.empty:
                res = validation_df[val_cols].copy()
                res.columns = val_headers
                res.to_excel(writer, sheet_name="LOG", index=False)
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
