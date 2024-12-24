import json
import logging

import pandas as pd
import requests


def read_endpoint(url):
    items = []
    logging.info(f"网络请求 {url} ...")
    try:
        page = 1
        size = 100
        while True:
            response = requests.get(f"{url}&current={page}&size={size}", timeout=60)
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get("success"):
                    data = json_data.get("data", [])
                    items.extend(data.get("data", []))
                    if page >= data.get('total_pages', 1):
                        break
                    page += 1
                else:
                    raise RuntimeError(f"网络请求 {url} 失败, {json_data.get('message')}")
            else:
                raise RuntimeError(f"网络请求 {url} 失败, code {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"网络请求 {url} 失败, {e}")

    df = pd.DataFrame(items)
    if len(df) == 0:
        raise RuntimeError(f"网络请求 {url} 失败, 无数据")

    logging.info(f"网络请求 {url} 成功。 数量为: {len(items)}")
    return df


def process_order(df, months=12):
    logging.info(f"处理订单数据")

    # 转换日期列
    df["order_date"] = pd.to_datetime(df["order_date"])

    # 获取当前日期，并计算开始日期
    current_date = pd.Timestamp.now().normalize()
    start_date = current_date - pd.DateOffset(months=months)

    # 确保金额列是数值类型
    df["order_amount_tax_included"] = pd.to_numeric(df["order_amount_tax_included"], errors="coerce")

    # 筛选最近的订单
    df_recent = df[(df["order_date"] >= start_date) & (df["order_date"] <= current_date)]

    # 按供应商名称计算平均订单金额
    avg_df_recent = df_recent.groupby("supplier_name")["order_amount_tax_included"].mean().reset_index()
    # avg_df_recent["order_amount_tax_included"] = avg_df_recent["order_amount_tax_included"].round(2)
    avg_df_recent.rename(columns={"order_amount_tax_included": "avg_order_amount"}, inplace=True)

    logging.info(f"处理订单数据成功。数量为: {len(avg_df_recent)}")
    return avg_df_recent


def process_model(order_df, supplier_df, model_df, supplier):
    logging.info(f"两方处理数据")
    if 'order_date' not in order_df.columns:
        raise RuntimeError("order_date is not in order file")
    if 'order_amount_tax_included' not in order_df.columns:
        raise RuntimeError("order_amount_tax_included is not in order file")
    if 'supplier_name' not in order_df.columns:
        raise RuntimeError("supplier_name is not in order file")

    if 'supplier_name' not in supplier_df.columns:
        raise RuntimeError("supplier_name is not in supplier file")
    if 'purchaser_name' not in supplier_df.columns:
        raise RuntimeError("purchaser_name is not in supplier file")
    if 'cooperation_duration' not in supplier_df.columns:
        raise RuntimeError("cooperation_duration is not in supplier file")
    if 'latest_rating' not in supplier_df.columns:
        raise RuntimeError("latest_rating is not in supplier file")

    if 'cooperation_duration' not in model_df.columns:
        raise RuntimeError("cooperation_duration is not in model file")
    if 'latest_rating' not in model_df.columns:
        raise RuntimeError("latest_rating is not in model file")
    if 'avg_payment_cycle' not in model_df.columns:
        raise RuntimeError("avg_payment_cycle is not in model file")

    cooperation_duration = float(model_df.iloc[0]["cooperation_duration"])
    latest_rating = float(model_df.iloc[0]["latest_rating"])
    avg_payment_cycle = float(model_df.iloc[0]["avg_payment_cycle"])
    months = 12
    if 'months' in model_df.columns:
        months = int(model_df.iloc[0]["months"])

    if supplier:
        order_df = order_df[order_df["supplier_name"].isin(supplier)]
        supplier_df = supplier_df[supplier_df["supplier_name"].isin(supplier)]
    order_df_processed = process_order(order_df, months=months)
    result_df = supplier_df.merge(order_df_processed, on="supplier_name", how="left")

    data = []
    effective_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=12)
    for _, row in result_df.iterrows():
        y = avg_payment_cycle
        x = 0.7
        if row['cooperation_duration'] > cooperation_duration:
            x = 0.9
        if row['latest_rating'] > latest_rating:
            x = 0.9
        financing_limit = round(row['avg_order_amount'] * x * y, 2)
        data.append({
            "supplier_name": row["supplier_name"],
            "core_enterprise_name": row['purchaser_name'] if row['purchaser_name'] else "",
            "financing_limit": financing_limit,
            "limit_effective_date": effective_date.strftime('%Y-%m-%d'),
            "status": True,
            "cooperating_bank": "浙商银行"
        })

    result_df = pd.DataFrame(data, columns=["supplier_name", "core_enterprise_name", "financing_limit", "limit_effective_date", "status", "cooperating_bank"])
    logging.info(f"两方处理数据成功 {len(result_df)}")
    return result_df


def save_ori_file(df, path, feature, url, task_id):
    if feature:
        df = df[feature]
    df.to_csv(path, index=False)
    if url:
        logging.info(f"网络请求 {url} ...")
        try:
            params = json.loads(df.to_json(orient="records"))
            payload = {
                'task_id': task_id,
                "params": params
            }
            logging.info(f"网络请求 {url} ... {payload}")
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                logging.info(f"网络请求 {url} 成功")
            else:
                raise RuntimeError(f"网络请求 {url} 失败, code {response.status_code}")
        except Exception as e:
            raise RuntimeError(f"网络请求 {url} 失败, {e}")
    return df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
        handlers=[logging.StreamHandler()]  # 将日志输出到终端
    )

    data_endpoint = "http://10.1.120.42:8080"
    rule_endpoint = "http://10.1.120.42:8080"
    logging.info(f"读取订单数据")
    order_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=order")
    logging.info(f"读取订单数据成功")

    order_df["supplier_name"] = order_df["supplier_name"].replace("测试供应商01", "测试供应商001")

    logging.info(f"读取供应商数据")
    supplier_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=supplier")
    logging.info(f"读取供应商数据成功")

    logging.info(f"读取模型数据")
    model_df = read_endpoint(f"{rule_endpoint}/tmpc/model/params/?type=credit_limit")
    logging.info(f"读取模型数据成功")

    logging.info(f"联合处理数据")
    result_df = process_model(order_df, supplier_df, model_df, [''])
    logging.info(f"联合处理数据成功")

    save_ori_file(result_df, "model_available.csv", None, f"{data_endpoint}/tmpc/model/update/?type=credit_limit",
                  'credit_limit')
