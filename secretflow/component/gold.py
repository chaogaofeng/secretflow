import pandas as pd
import logging
from jax import numpy as jnp
from secretflow.error_system import CompEvalError


def read_file(filepath, columns=None):
    logging.info(f"读取文件 {filepath} ...")
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except Exception as e:
        df = pd.read_csv(filepath, encoding="gbk")
    logging.info(f"读取文件完成 {filepath}。数量为: {len(df)}")

    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise CompEvalError(f"以下列在数据中缺失: {missing_columns}")

    return df


def save_file(path, df, columns=None):
    logging.info(f"保存文件 {path} ...")
    df = df[columns]
    df.to_csv(path, index=False)
    logging.info(f"保存文件完成 {path}。")
    id_types = [
        "str" if df[col].dtype == "object" else "float"
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


def prepare_data_by_supplier(data_order_df, data_supplier_df, columns, supplier=[], months=12):
    """
    准备数据
    """
    logging.info(f"数据预处理...")
    # 筛选供应商
    if supplier:
        data_order_df = data_order_df[data_order_df["supplier_name"].isin(supplier)]
        data_supplier_df = data_supplier_df[data_supplier_df["supplier_name"].isin(supplier)]

    # 确保关键列存在
    required_columns = ["supplier_name", "order_date", "order_amount_tax_included"]
    missing_columns = [col for col in required_columns if col not in data_order_df.columns]
    if missing_columns:
        raise CompEvalError(f"以下列在数据中缺失: {missing_columns}")

    # 转换日期列
    data_order_df["order_date"] = pd.to_datetime(data_order_df["order_date"])
    data_order_df["order_amount_tax_included"] = pd.to_numeric(data_order_df["order_amount_tax_included"],
                                                               errors="coerce")
    # 去除无效数据
    data_order_df = data_order_df.dropna(subset=["order_date", "order_amount_tax_included"])
    # 获取当前日期，并计算开始日期
    current_date = pd.Timestamp.now().normalize()
    start_date = current_date - pd.DateOffset(months=months)
    # 筛选最近的订单
    df_recent = data_order_df[
        (data_order_df["order_date"] >= start_date) & (data_order_df["order_date"] <= current_date)]
    # 提取发生订单的月份
    df_recent['order_month'] = df_recent['order_date'].dt.to_period('M')
    # 确保金额列是数值类型
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

    df = data_supplier_df.merge(processed_df, on="supplier_name", how="left")
    df.fillna({"total_order_amount": 0, "avg_order_amount": 0}, inplace=True)

    new_df = df[columns]
    np_data = new_df.to_numpy()
    np_columns = {col: i for i, col in enumerate(new_df.columns)}
    logging.info(f'数据预处理完成。数量为: {len(df)}')
    return df, np_data, np_columns


def prepare_data_by_order(data_order_df, data_receipt_df, data_invoice_df, data_voucher_df, columns, order=[]):
    """
    准备数据
    """
    logging.info(f"数据预处理...")

    if order:
        data_order_df = data_order_df[data_order_df["order_number"].isin(order)]
        data_receipt_df = data_receipt_df[data_receipt_df["order_number"].isin(order)]
        data_invoice_df = data_invoice_df[data_invoice_df["order_number"].isin(order)]
        data_voucher_df = data_voucher_df[data_voucher_df["order_number"].isin(order)]

    df = data_order_df.merge(data_invoice_df[["order_number", "total_amount_with_tax"]], on="order_number", how="left")
    df = df.merge(data_receipt_df[["order_number", "receipt_amount_tax_included"]], on="order_number", how="left")
    df = df.merge(data_voucher_df[["order_number", "credit_amount"]], on="order_number", how="left")
    df["credit_amount"] = pd.to_numeric(df["credit_amount"], errors="coerce")
    df["order_amount_tax_included"] = pd.to_numeric(df["order_amount_tax_included"], errors="coerce")
    df["total_amount_with_tax"] = pd.to_numeric(df["total_amount_with_tax"], errors="coerce")
    df.fillna(
        {"credit_amount": 0, "order_amount_tax_included": 0, "total_amount_with_tax": 0, "total_amount_with_tax": 0},
        inplace=True)

    new_df = df[columns]
    np_data = new_df.to_numpy()
    np_columns = {col: i for i, col in enumerate(new_df.columns)}
    logging.info(f'数据预处理完成。数量为: {len(df)}')
    return df, np_data, np_columns


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

    return jnp.where(condition, True, False)


def processed_marketing(df, result):
    import numpy as np
    df["is_qualified"] = np.where(result, "是", "否")
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


def processed_quota(df, result):
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


def processed_withdraw(df, result):
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


def processed_monitoring(df, result, months=12):
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


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
        handlers=[logging.StreamHandler()]  # 将日志输出到终端
    )
    import secretflow as sf

    # sf.shutdown()
    sf.init(parties=['alice', 'bob', 'carol'], address='local')
    alice, bob = sf.PYU('alice'), sf.PYU('bob')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    data_dir = '/root/workspace/secretflow2/secretflow/component/poc/'
    data_order_df = alice(read_file)(data_dir + '订单表.csv',
                                     ['purchaser_name', 'supplier_name', 'order_number'])
    data_supplier_df = alice(read_file)(data_dir + '供应商信息表.csv',
                                        ['supplier_name', 'purchaser_name', 'cooperation_duration', 'latest_rating'])
    data_receipt_df = alice(read_file)(data_dir + '入库表.csv', ["order_number", "receipt_amount_tax_included"])
    data_invoice_df = alice(read_file)(data_dir + '发票表.csv', ["order_number", "total_amount_with_tax"])
    data_voucher_df = alice(read_file)(data_dir + '应收应付.csv', ["order_number", "credit_amount"])

    # model_df = bob(read_file)(data_dir + '营销模型.csv')
    # model_df = bob(read_file)(data_dir + '额度模型.csv', ['cooperation_duration', 'latest_rating', 'avg_payment_cycle'])
    # model_df = bob(read_file)(data_dir + '贷后检测模型.csv')
    model_df = bob(read_file)(data_dir + '提款模型.csv',
                              ['financing_balance_param', 'delivered_uninvoiced_amount_param',
                               'undelivered_amount_param'])

    df_pyu_obj, np_data_pyu_obj, np_column_pyu_obj = alice(prepare_data_by_order, num_returns=3)(data_order_df,
                                                                                                 data_receipt_df,
                                                                                                 data_invoice_df,
                                                                                                 data_voucher_df,
                                                                                                 [
                                                                                                     'credit_amount',
                                                                                                     'order_amount_tax_included',
                                                                                                     'total_amount_with_tax'],
                                                                                                 ['H20240103001'])
    params_pyu_obj = bob(prepare_params)(model_df)

    from secretflow.device import SPUCompilerNumReturnsPolicy

    np_data_spu_object = np_data_pyu_obj.to(spu)
    np_column_spu_obj = np_column_pyu_obj.to(spu)
    params_spu_object = params_pyu_obj.to(spu)
    ret_spu_obj = spu(
        process_withdraw,
        num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
        user_specified_num_returns=1,
    )(np_data_pyu_obj, np_column_spu_obj, params_spu_object)

    ret_pyu_obj = ret_spu_obj.to(alice)
    result_df = alice(processed_withdraw)(df_pyu_obj, ret_pyu_obj)
    logging.info(f"market result_df: {sf.reveal(result_df)}")

    features = [
        "supplier_name", "core_enterprise_name", "order_number", "order_amount",
        "financing_amount", "application_date", "status", "approved_financing_amount"
    ]

    output_data = "withdraw_result"
    output_data_path = f"{alice}.csv"
    logging.info(f"数据方输出文件")
    output_data_types = alice(save_file)(output_data_path, result_df, features)
    logging.info(f"数据方输出输出文件成功")

    output_rule_path = f"{bob}.csv"
    logging.info(f"规则方输出文件")
    result_df = result_df.to(bob)
    output_rule_types = bob(save_file)(output_rule_path, result_df, features)
    logging.info(f"规则方输出文件成功")

    logging.info("组件输出结果")
    from secretflow.spec.v1.data_pb2 import (
        DistData,
        IndividualTable,
        StorageConfig,
        TableSchema,
    )
    from secretflow.component.data_utils import (
        DistDataType,
        extract_data_infos,
    )

    # generate DistData
    output_data_db = DistData(
        name="ddd",
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=output_data_path, party=str(alice), format="csv")],
    )

    output_data_types = sf.reveal(output_data_types)
    output_rule_types = sf.reveal(output_rule_types)
    print(output_rule_types)
    output_data_meta = IndividualTable(
        schema=TableSchema(
            id_types=output_data_types,
            ids=features,
        ),
        line_count=-1,
    )
    output_data_db.meta.Pack(output_data_meta)

    output_rule_db = DistData(
        name='ddd',
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=output_rule_path, party=str(bob), format="csv")],
    )
    output_rule_meta = IndividualTable(
        schema=TableSchema(
            id_types=output_rule_types,
            ids=features,
        ),
        line_count=-1,
    )
    output_rule_db.meta.Pack(output_rule_meta)
