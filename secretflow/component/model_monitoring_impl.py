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

    # 按供应商分组计算累计金额
    processed_df = df_recent.groupby("supplier_name")["order_amount_tax_included"].sum().reset_index()
    processed_df.rename(columns={"order_amount_tax_included": f"total_order_amount"}, inplace=True)

    logging.info(f"处理订单数据成功。数量为: {len(processed_df)}")
    return processed_df


def process_model(order_df, supplier_df, model_df):
    logging.info(f"两方处理数据")
    if 'order_date' not in order_df.columns:
        raise RuntimeError("order_date is not in order file")
    if 'order_amount_tax_included' not in order_df.columns:
        raise RuntimeError("order_amount_tax_included is not in order file")
    if 'supplier_name' not in order_df.columns:
        raise RuntimeError("supplier_name is not in order file")

    if 'supplier_name' not in supplier_df.columns:
        raise RuntimeError("supplier_name is not in supplier file")
    # if "is_qualified" not in supplier_df.columns:
    #     raise RuntimeError("is_qualified is not in supplier file")
    # if 'cooperation_duration' not in supplier_df.columns:
    #     raise RuntimeError("cooperation_duration is not in supplier file")
    if 'latest_rating' not in supplier_df.columns:
        raise RuntimeError("latest_rating is not in supplier file")

    # if 'cooperation_duration' not in model_df.columns:
    #     raise RuntimeError("cooperation_duration is not in model file")
    if 'latest_rating' not in model_df.columns:
        raise RuntimeError("latest_rating is not in model file")
    if 'total_order_amount' not in model_df.columns:
        raise RuntimeError("total_order_amount is not in model file")

    # cooperation_duration = float(model_df.iloc[0]["cooperation_duration"])
    latest_rating = float(model_df.iloc[0]["latest_rating"])
    total_order_amount = float(model_df.iloc[0]["total_order_amount"])
    months = 12
    if 'months' in model_df.columns:
        months = int(model_df.iloc[0]["months"])
    order_df_processed = process_order(order_df, months=months)
    result_df = supplier_df.merge(order_df_processed, on="supplier_name")
    # result_df["warning_status"] = result_df.apply(lambda x: True if (
    #         x["latest_rating"] < latest_rating or
    #         x["total_order_amount"] < total_order_amount) else False,
    #                                             axis=1)

    monitoring_data = []
    for _, row in result_df.iterrows():
        # 添加供应商评分监测
        if row["latest_rating"] < latest_rating:
            monitoring_data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": "供应商评分",
                "monitoring_value": row["latest_rating"],
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": row['contact_person'] if 'contact_person' in row else ""
            })
        elif row["total_order_amount"] < total_order_amount:
            monitoring_data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": f"供应商近{months}个月与核企累计订单金额",
                "monitoring_value": row["latest_rating"],
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": row['contact_person'] if 'contact_person' in row else ""
            })
        else:
            monitoring_data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": "",
                "monitoring_value": "",
                "warning_status": False,
                "warning_method": "",
                "receiver": row['contact_person'] if 'contact_person' in row else ""
            })

    result_df = pd.DataFrame(monitoring_data)
    logging.info(f"两方处理数据成功 {len(result_df)}")
    return result_df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
        handlers=[logging.StreamHandler()]  # 将日志输出到终端
    )

    data_endpoint = "http://10.1.120.42:8070"
    rule_endpoint = "http://10.1.120.42:8070"
    logging.info(f"读取订单数据")
    order_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=order")
    logging.info(f"读取订单数据成功")

    order_df["supplier_name"] = order_df["supplier_name"].replace("测试供应商01", "测试供应商001")

    logging.info(f"读取供应商数据")
    supplier_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=supplier")
    logging.info(f"读取供应商数据成功")

    logging.info(f"读取模型数据")
    model_df = read_endpoint(f"{rule_endpoint}/tmpc/model/params/?type=loan_follow_up")
    logging.info(f"读取模型数据成功")

    logging.info(f"联合处理数据")
    result_df = process_model(order_df, supplier_df, model_df)
    logging.info(f"联合处理数据成功")

    logging.info(result_df)
    