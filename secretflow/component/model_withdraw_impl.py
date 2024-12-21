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
    avg_df_recent.rename(columns={"order_amount_tax_included": "avg_order_amount"}, inplace=True)

    logging.info(f"处理订单数据成功。数量为: {len(avg_df_recent)}")
    return avg_df_recent


def process_model(order_df, invoice_df, receipt_df, voucher_df, model_df, order_number):
    logging.info(f"两方处理数据")
    if 'order_number' not in order_df.columns:
        raise RuntimeError("order_number is not in order file")
    if 'order_amount_tax_included' not in order_df.columns:
        raise RuntimeError("order_amount_tax_included is not in order file")
    if 'supplier_name' not in order_df.columns:
        raise RuntimeError("supplier_name is not in order file")

    if 'contract_number' not in invoice_df.columns:
        raise RuntimeError("contract_number is not in invoice file")
    if 'total_amount_with_tax' not in invoice_df.columns:
        raise RuntimeError("total_amount_with_tax is not in invoice file")
    invoice_df.rename(columns={"contract_number": "order_number"}, inplace=True)

    if 'order_number' not in receipt_df.columns:
        raise RuntimeError("order_number is not in receipt file")
    if 'receipt_amount_tax_included' not in receipt_df.columns:
        raise RuntimeError("receipt_amount_tax_included is not in receipt file")

    if 'contract_number' not in voucher_df.columns:
        raise RuntimeError("contract_number is not in voucher file")
    if 'credit_amount' not in voucher_df.columns:
        raise RuntimeError("credit_amount is not in voucher file")
    voucher_df.rename(columns={"contract_number": "order_number"}, inplace=True)

    if 'financing_balance_param' not in model_df.columns:
        raise RuntimeError("financing_balance_param is not in model file")
    if 'delivered_uninvoiced_amount_param' not in model_df.columns:
        raise RuntimeError("delivered_uninvoiced_amount_param is not in model file")
    if 'undelivered_amount_param' not in model_df.columns:
        raise RuntimeError("undelivered_amount_param is not in model file")

    financing_balance_param = float(model_df.iloc[0]["financing_balance_param"])
    delivered_uninvoiced_amount_param = float(model_df.iloc[0]["delivered_uninvoiced_amount_param"])
    undelivered_amount_param = float(model_df.iloc[0]["undelivered_amount_param"])

    # TODO
    if order_number:
        order_df = order_df[order_df["order_number"].isin(order_number)]
        invoice_df = invoice_df[invoice_df["order_number"].isin(order_number)]
        receipt_df = receipt_df[receipt_df["order_number"].isin(order_number)]
        voucher_df = voucher_df[voucher_df["order_number"].isin(order_number)]
    result_df = order_df.merge(invoice_df, on="order_number", how="left")
    result_df = order_df.merge(voucher_df, on="order_number", how="left")
    result_df = order_df.merge(receipt_df, on="order_number", how="left")
    result_df = result_df.merge(order_df, on="order_number", how="left")

    # TODO
    return result_df


def save_ori_file(df, path, feature, url, task_id):
    if feature:
        df = df[feature]
    df.to_csv(path, index=False)
    if url:
        logging.info(f"网络请求 {url} ...")
        try:
            payload = {
                'task_id': task_id,
                "params": df.to_json(orient="records")
            }
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

    data_endpoint = "http://10.1.120.42:8070"
    rule_endpoint = "http://10.1.120.42:8070"
    logging.info(f"读取订单数据")
    order_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=order")
    logging.info(f"读取订单数据成功")

    logging.info(f"读取发票数据")
    invoice_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=invoice")
    logging.info(f"读取发票数据成功")

    logging.info(f"读取入库数据")
    receipt_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=warehouse_receipt")
    logging.info(f"读取入库数据成功")

    logging.info(f"读取应付数据")
    voucher_df = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=voucher")
    logging.info(f"读取应付数据成功")

    logging.info(f"读取模型数据")
    model_df = read_endpoint(f"{rule_endpoint}/tmpc/model/params/?type=credit_limit")
    logging.info(f"读取模型数据成功")

    logging.info(f"联合处理数据")
    result_df = process_model(order_df, invoice_df, receipt_df, voucher_df, model_df, [])
    logging.info(f"联合处理数据成功")

    save_ori_file(result_df, "model_withdraw.csv", None, f"{data_endpoint}/tmpc/model/update/?type=financing_application",
                  'financing_application')
