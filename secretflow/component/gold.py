import pandas as pd

from secretflow.utils import logging
from jax import numpy as jnp


def read_file(filepath, columns=None):
    logging.info(f"读取文件 {filepath} ...")
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except Exception as e:
        df = pd.read_csv(filepath, encoding="gbk")
    logging.info(f"读取文件完成 {filepath}。数量为: {len(df)}")

    if columns:
        from secretflow.error_system import CompEvalError
        for column in columns:
            if column not in df.columns:
                raise CompEvalError(f"{column} 不在数据列中")
    return df


def save_file(path, df, columns=None):
    logging.info(f"保存文件 {path} ...")
    df = df[columns]
    df.to_csv(path, index=False)
    logging.info(f"保存文件完成 {path}。")
    id_types = [
        "str" if df[col].dtype == "object" else "int"
        for col in columns
    ]
    return id_types


def prepare_params(df):
    """
    规则参数
    """
    logging.info(f"模型参数预处理...")
    df = df.astype(float)
    params = df.to_dict(orient='records')[0]
    logging.info(f"模型参数预处理完成。")
    return params


def prepare_data_by_supplier(data_order_df, data_supplier_df, supplier=[], months=12):
    """
    准备数据
    """
    logging.info(f"数据预处理...")
    # 筛选供应商
    if supplier:
        data_order_df = data_order_df[data_order_df["supplier_name"].isin(supplier)]
        data_supplier_df = data_supplier_df[data_supplier_df["supplier_name"].isin(supplier)]
    df = data_supplier_df.merge(data_order_df, on="supplier_name", how="left")
    # 转换日期列
    df["order_date"] = pd.to_datetime(df["order_date"])
    # 获取当前日期，并计算开始日期
    current_date = pd.Timestamp.now().normalize()
    start_date = current_date - pd.DateOffset(months=months)
    # 筛选最近的订单
    df_recent = df[(df["order_date"] >= start_date) & (df["order_date"] <= current_date)]
    # 提取发生订单的月份
    df_recent['order_month'] = df_recent['order_date'].dt.to_period('M')
    # 确保金额列是数值类型
    df["order_amount_tax_included"] = pd.to_numeric(df["order_amount_tax_included"], errors="coerce")
    df_monthly = df_recent.groupby(["supplier_name", "order_month"]).agg(
        monthly_order_amount=("order_amount_tax_included", "sum")
    ).reset_index()

    # 按供应商名称计算实际发生月份的平均订单金额
    processed_df = df_monthly.groupby("supplier_name").agg(
        total_order_amount=("monthly_order_amount", "sum"),
        avg_order_amount=("monthly_order_amount", "mean")
    ).reset_index()

    # 保留小数位两位
    processed_df["total_order_amount"] = processed_df["total_order_amount"].round(2)
    processed_df["avg_order_amount"] = processed_df["avg_order_amount"].round(2)
    np_data = df.to_numpy()
    np_columns = {col: i for i, col in enumerate(df.columns)}
    logging.info(f'数据预处理完成。数量为: {len(df)}')
    return np_data, np_columns


def prepare_data_by_order(df, order=[]):
    """
    准备数据
    """
    logging.info(f"数据预处理...")

    if order:
        df = df[df["order_number"].isin(order)]
    df["credit_amount"] = pd.to_numeric(df["credit_amount"], errors="coerce")
    df["order_amount_tax_included"] = pd.to_numeric(df["order_amount_tax_included"], errors="coerce")
    df["total_amount_with_tax"] = pd.to_numeric(df["total_amount_with_tax"], errors="coerce")
    df.fillna(
        {"credit_amount": 0, "order_amount_tax_included": 0, "total_amount_with_tax": 0, "total_amount_with_tax": 0},
        inplace=True)

    np_data = df.to_numpy()
    np_columns = {col: i for i, col in enumerate(df.columns)}
    logging.info(f'数据预处理完成。数量为: {len(df)}')
    return np_data, np_columns


def process_marketing(np_data, np_columns, params):
    """
    供应商合作时长>=3年
    供应商评分>=70
    近12个月与核企累计订单金额>200万
    """
    condition1 = np_data[:, np_columns['cooperation_duration']] >= params['cooperation_duration']  # 合作时长条件
    condition2 = np_data[:, np_columns['latest_rating']] >= params['latest_rating']  # 最新评分条件
    condition3 = np_data[:, np_columns['total_order_amount']] > params['total_order_amount']  # 总订单金额条件
    # 合并条件
    condition = condition1 & condition2 & condition3

    return jnp.where(condition, '是', '否')


def processed_marketing(np_data, np_columns, result):
    df = pd.DataFrame(np_data, columns=np_columns.keys())
    df["is_qualified"] = result
    return df


def process_quota(np_data, np_columns, params):
    """
    供应商可融资额度=近12个月平均发票金额*X*Y
    X取值逻辑为：if 合作时长>5年 then X=0.9
    if 供应商评分>=90 then X=0.9 ELSE X=0.7
    Y取值逻辑为供应商平均回款周期，假定6个月，则Y=6
    """
    condition_1 = (np_data[:, np_columns['cooperation_duration']] > params['cooperation_duration'])
    condition_2 = (np_data[:, np_columns['latest_rating']] >= params['latest_rating'])

    result = jnp.where(condition_1, 0.9, jnp.where(condition_2, 0.9, 0.7)) * 100 * params['avg_payment_cycle']
    return result


def processed_quota(np_data, np_columns, result):
    df = pd.DataFrame(np_data, columns=np_columns.keys())

    data = []
    effective_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=12)
    for index, row in df.iterrows():
        financing_limit = round(result[index] / 100 * row["avg_order_amount"], 2)
        data.append({
            "supplier_name": row["supplier_name"],
            "core_enterprise_name": row['purchaser_name'] if row['purchaser_name'] else "",
            "financing_limit": financing_limit,
            "limit_effective_date": effective_date.strftime('%Y-%m-%d'),
            "status": True,
            "cooperating_bank": "浙商银行"
        })

    new_df = pd.DataFrame(data, columns=["supplier_name", "core_enterprise_name", "financing_limit",
                                         "limit_effective_date", "status", "cooperating_bank"])
    return new_df


def process_withdraw(np_data, np_columns, params):
    """
    可提款金额=融资合同应付余额*90%+已开票未挂账金额*70%
    融资合同号H20240103，可提款金额=572232*0.9+0*0.7+0*0.4
    """
    result = np_data[:, np_columns['credit_amount']] * params['financing_balance_param'] + (
            np_data[:, np_columns['order_amount_tax_included']] - np_data[:, np_columns['total_amount_with_tax']]) * \
             params[
                 'delivered_uninvoiced_amount_param'] + (
                     np_data[:, np_columns['credit_amount']] - np_data[:, np_columns['total_amount_with_tax']]) * \
             params['undelivered_amount_param']
    return result


def processed_withdraw(np_data, np_columns, result):
    df = pd.DataFrame(np_data, columns=np_columns.keys())

    data = []
    effective_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=12)
    for index, row in df.iterrows():
        financing_limit = round(result[index], 2)
        data.append({
            "supplier_name": row["supplier_name"],
            "core_enterprise_name": row['purchaser_name'] if row['purchaser_name'] else "",
            "order_number": row["order_number"],
            "order_amount": row["order_amount_tax_included"],
            "financing_amount": "",
            "application_date": "",
            "status": "已批准",
            "approved_financing_amount": financing_limit if financing_limit > 0 else 0,
        })

    new_df = pd.DataFrame(data, columns=["supplier_name", "core_enterprise_name", "order_number", "order_amount",
                                         "financing_amount", "application_date", "status", "approved_financing_amount"])
    return new_df


def process_monitoring(np_data, np_columns, params):
    """
    if 供应商评分<70 then 消息提醒平台运营 王五
    if 供应商近12个月与核企累计发票金额<200万 then 消息提醒平台运营 王五
    """
    condition1 = (np_data[:, np_columns['latest_rating']] < params['latest_rating'])
    condition2 = (np_data[:, np_columns['total_order_amount']] < params['total_order_amount'])

    return jnp.where(
        condition1 & condition2, 3,
        jnp.where(condition1, 1, jnp.where(condition2, 2, 0))
    )


def processed_monitoring(np_data, np_columns, result, months=12):
    df = pd.DataFrame(np_data, columns=np_columns.keys())

    data = []
    for index, row in df.iterrows():
        # 添加供应商评分监测
        if result[index] == 3:
            data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": f"供应商评分, 供应商近{months}个月与核企累计订单金额",
                "monitoring_value": ','.join([str(row["latest_rating"]), f'{row["total_order_amount"]:,.2f}']),
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": "王五"
            })
        if result[index] == 1:
            data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": "供应商评分",
                "monitoring_value": ','.join([str(row["latest_rating"])]),
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": "王五"
            })
        elif result[index] == 2:
            data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": f"供应商近{months}个月与核企累计订单金额",
                "monitoring_value": ','.join([f'{row["total_order_amount"]:,.2f}']),
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": "王五"
            })
    new_df = pd.DataFrame(data,
                          columns=["supplier_name", "monitoring_item", "monitoring_value", "warning_status",
                                   "warning_method", "receiver"])
    return new_df
