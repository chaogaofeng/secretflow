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


def process_model(order_df, invoice_df, receipt_df, voucher_df, model_df, order_number):
    logging.info(f"两方处理数据")
    if 'order_number' not in order_df.columns:
        raise RuntimeError("order_number is not in order file")
    if 'order_amount_tax_included' not in order_df.columns:
        raise RuntimeError("order_amount_tax_included is not in order file")
    if 'supplier_name' not in order_df.columns:
        raise RuntimeError("supplier_name is not in order file")
    order_df["order_amount_tax_included"] = pd.to_numeric(order_df["order_amount_tax_included"], errors="coerce")

    if 'order_number' not in invoice_df.columns:
        raise RuntimeError("order_number is not in invoice file")
    if 'total_amount_with_tax' not in invoice_df.columns:
        raise RuntimeError("total_amount_with_tax is not in invoice file")
    invoice_df["total_amount_with_tax"] = pd.to_numeric(invoice_df["total_amount_with_tax"], errors="coerce")

    if 'order_number' not in receipt_df.columns:
        raise RuntimeError("order_number is not in receipt file")
    if 'receipt_amount_tax_included' not in receipt_df.columns:
        raise RuntimeError("receipt_amount_tax_included is not in receipt file")
    receipt_df["receipt_amount_tax_included"] = pd.to_numeric(receipt_df["receipt_amount_tax_included"],
                                                              errors="coerce")

    if 'order_number' not in voucher_df.columns:
        raise RuntimeError("order_number is not in voucher file")
    if 'credit_amount' not in voucher_df.columns:
        raise RuntimeError("credit_amount is not in voucher file")
    voucher_df["credit_amount"] = pd.to_numeric(voucher_df["credit_amount"], errors="coerce")

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
    result_df = order_df.merge(invoice_df[["order_number", "total_amount_with_tax"]], on="order_number", how="left")
    result_df = result_df.merge(voucher_df[["order_number", "credit_amount"]], on="order_number", how="left")
    result_df = result_df.merge(receipt_df[["order_number", "receipt_amount_tax_included"]], on="order_number",
                                how="left")
    result_df.fillna(
        {"credit_amount": 0, "order_amount_tax_included": 0, "total_amount_with_tax": 0, "total_amount_with_tax": 0},
        inplace=True)
    # result_df['approved_financing_amount'] = (result_df['credit_amount'] * 0.9
    #                                           + (result_df['order_amount_tax_included'] - result_df['total_amount_with_tax'])*0.7
    #                                           +(result_df['credit_amount'] - result_df['total_amount_with_tax'])*0.4).round(2)
    data = []
    for _, row in result_df.iterrows():
        approved_financing_amount = round(row['credit_amount'] * financing_balance_param
                                          + (row['order_amount_tax_included'] - row['total_amount_with_tax']) * delivered_uninvoiced_amount_param
                                          + (row['credit_amount'] - row['total_amount_with_tax']) * undelivered_amount_param, 2)
        data.append({
            "supplier_name": row["supplier_name"],
            "core_enterprise_name": row['purchaser_name'] if row['purchaser_name'] else "",
            "order_number": row["order_number"],
            "order_amount": row["order_amount_tax_included"],
            "financing_amount": "0",
            "application_date": "2024-12-09",
            "status": "",
            "approved_financing_amount": approved_financing_amount,
        })

    result_df = pd.DataFrame(data,
                             columns=["supplier_name", "core_enterprise_name", "order_number", "order_amount",
                                      "financing_amount", "application_date", "status", "approved_financing_amount"])
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
    order_number = ['H20240103001']

    data_endpoint = "http://10.1.120.42:8080"
    rule_endpoint = "http://10.1.120.42:8080"
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
    model_df = read_endpoint(f"{rule_endpoint}/tmpc/model/params/?type=financing_application")
    logging.info(f"读取模型数据成功")

    logging.info(f"联合处理数据")
    result_df = process_model(order_df, invoice_df, receipt_df, voucher_df, model_df, order_number)
    logging.info(f"联合处理数据成功")

    save_ori_file(result_df, "model_withdraw.csv", None,
                  f"{data_endpoint}/tmpc/model/update/?type=financing_application",
                  'financing_application')
