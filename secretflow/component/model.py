import jax
import pandas as pd
import logging
import requests


def read_endpoint(url):
    """
    读取指定 URL 的数据，并返回 DataFrame
    """
    items = []
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
                    raise RuntimeError(f"{json_data.get('message')}")
            else:
                raise RuntimeError(f"code != {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"网络请求 {url} 失败, {e}")

    df = pd.DataFrame(items)
    if len(df) == 0:
        raise RuntimeError(f"网络请求 {url} 失败, 无数据")
    return df


def process_order_by_supplier(df, months=12):
    """
    按照供应商统计订单总金额、数量、平均金额
    """

    logging.info(f"处理订单数据...")
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
    processed_df = df_recent.groupby("supplier_name").agg(
        total_order_amount=("order_amount_tax_included", "sum"),
        total_order_count=("order_amount_tax_included", "count")
    ).reset_index()

    # 计算平均订单金额
    processed_df["avg_order_amount"] = processed_df["total_order_amount"] / processed_df["total_order_count"]
    # 保留小数位两位
    processed_df["total_order_amount"] = processed_df["total_order_amount"].round(2)
    processed_df["avg_order_amount"] = processed_df["avg_order_amount"].round(2)

    logging.info(f"处理订单数据完毕。数量为: {len(processed_df)}")

    return processed_df


def prepare_data_by_supplier(endpoint, columns, supplier=[]):
    """
    准备数据
    """
    logging.info(f'获取订单数据...')
    df_order = read_endpoint(f"{endpoint}/tmpc/data/list/?type=order")
    if supplier:
        df_order = df_order[df_order["supplier_name"].isin(supplier)]
    logging.info(f'获取订单数据完毕。数量为: {len(df_order)}')

    logging.info(f'获取供应商数据...')
    df_supplier = read_endpoint(f"{endpoint}/tmpc/data/list/?type=supplier")
    if supplier:
        df_supplier = df_supplier[df_supplier["supplier_name"].isin(supplier)]
    logging.info(f'获取供应商数据完毕。数量为: {len(df_supplier)}')

    df_order_processed = process_order_by_supplier(df_order)
    df = df_supplier.merge(df_order_processed, on="supplier_name", how="left")
    df.fillna({"total_order_amount": 0, "total_order_count": 0, "avg_order_amount": 0}, inplace=True)

    logging.info(f'数据准备完毕。数量为: {len(df)}')
    new_columns = ['id']
    new_columns.extend(columns)
    # for column in columns:
    #     if column in df.columns:
    #         new_columns.append(column)
    new_df = df[new_columns]
    return df, new_df.to_numpy()


def prepare_data_by_order(endpoint, columns, order=[]):
    """
    准备数据
    """
    logging.info(f'获取订单数据...')
    df_order = read_endpoint(f"{endpoint}/tmpc/data/list/?type=order")
    if order:
        df_order = df_order[df_order["order_number"].isin(order)]
    logging.info(f'获取订单数据完毕。数量为: {len(df_order)}')

    logging.info(f'获取供应商数据...')
    df_supplier = read_endpoint(f"{endpoint}/tmpc/data/list/?type=supplier")
    logging.info(f'获取供应商数据完毕。数量为: {len(df_supplier)}')

    logging.info(f"读取发票数据")
    df_invoice = read_endpoint(f"{endpoint}/tmpc/data/list/?type=invoice")
    if order:
        df_invoice = df_invoice[df_invoice["order_number"].isin(order)]
    logging.info(f"读取发票数据完毕。数量为: {len(df_invoice)}")

    logging.info(f"读取入库数据")
    df_receipt = read_endpoint(f"{endpoint}/tmpc/data/list/?type=warehouse_receipt")
    if order:
        df_receipt = df_receipt[df_receipt["order_number"].isin(order)]
    logging.info(f"读取入库数据完毕。数量为: {len(df_receipt)}")

    logging.info(f"读取应付数据")
    df_voucher = read_endpoint(f"{endpoint}/tmpc/data/list/?type=voucher")
    if order:
        df_voucher = df_voucher[df_voucher["order_number"].isin(order)]
    logging.info(f"读取应付数据完毕。数量为: {len(df_voucher)}")

    df = df_order.merge(df_invoice[["order_number", "total_amount_with_tax"]], on="order_number", how="left")
    df = df.merge(df_receipt[["order_number", "receipt_amount_tax_included"]], on="order_number", how="left")
    df = df.merge(df_voucher[["order_number", "credit_amount"]], on="order_number", how="left")
    df["credit_amount"] = pd.to_numeric(df["credit_amount"], errors="coerce")
    df["order_amount_tax_included"] = pd.to_numeric(df["order_amount_tax_included"], errors="coerce")
    df["total_amount_with_tax"] = pd.to_numeric(df["total_amount_with_tax"], errors="coerce")
    df.fillna(
        {"credit_amount": 0, "order_amount_tax_included": 0, "total_amount_with_tax": 0, "total_amount_with_tax": 0},
        inplace=True)

    logging.info(f'数据准备完毕。数量为: {len(df)}')
    new_columns = ['id']
    new_columns.extend(columns)
    # for column in columns:
    #     if column in df.columns:
    #         new_columns.append(column)
    new_df = df[new_columns]
    return df, new_df.to_numpy()


def prepare_params(endpoint, columns, tp):
    """
    准备参数
    """
    logging.info(f"读取模型参数")
    df_model = read_endpoint(f"{endpoint}/tmpc/model/params/?type={tp}")
    logging.info(f'读取模型参数。数量为: {len(df_model)}')
    new_columns = []
    new_columns.extend(columns)
    # for column in columns:
    #     if column in df_model.columns:
    #         new_columns.append(column)
    df_model[new_columns] = df_model[new_columns].astype(float)
    df_model = df_model[new_columns]
    return df_model.to_dict(orient='records')[0]


def process_marketing(df, params):
    """
    模型计算
    """
    condition = (
            (df[:, 1] >= params['cooperation_duration']) &
            (df[:, 2] >= params['latest_rating']) &
            (df[:, 3] >= params['total_order_amount'])
    )
    return jax.numpy.where(condition, True, False)


def processed_marketing(df, ret_column):
    """
    模型结果
    """
    df["is_qualified"] = ret_column
    return df


def process_available(df, params):
    """
    模型计算
    """
    condition_1 = (df[:, 1] != 0) & (df[:, 1] > params['cooperation_duration'])
    condition_2 = (df[:, 2] != 0) & (df[:, 2] >= params['latest_rating'])
    result = jax.numpy.where(condition_1, 0.9,
                             jax.numpy.where(condition_2, 0.9, 0.7)) * params['avg_payment_cycle'] * df[:, 3]
    return result


def processed_available(df, ret_column):
    """
    模型结果
    """
    data = []
    effective_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=12)
    for index, row in df.iterrows():
        financing_limit = round(ret_column[index], 2)
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


def process_withdraw(df, params):
    """
    模型计算
    """
    result = df[:, 1] * params['financing_balance_param'] + (df[:, 2] - df[:, 3]) * params[
        'delivered_uninvoiced_amount_param'] + (df[:, 1] - df[:, 3]) * params['undelivered_amount_param']
    return result


def processed_withdraw(df, ret_column):
    """
    模型结果
    """
    data = []
    effective_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=12)
    for index, row in df.iterrows():
        financing_limit = round(ret_column[index], 2)
        data.append({
            "supplier_name": row["supplier_name"],
            "core_enterprise_name": row['purchaser_name'] if row['purchaser_name'] else "",
            "order_number": row["order_number"],
            "order_amount": row["order_amount_tax_included"],
            "financing_amount": "0",
            "application_date": effective_date.strftime('%Y-%m-%d'),
            "status": "",
            "approved_financing_amount": financing_limit if financing_limit > 0 else 0,
        })

    new_df = pd.DataFrame(data, columns=["supplier_name", "core_enterprise_name", "order_number", "order_amount",
                                         "financing_amount", "application_date", "status", "approved_financing_amount"])
    return new_df


def process_monitoring(df, params):
    """
    模型计算
    """
    # condition = (
    #         ((df[:, 1] > 0) & (df[:, 1] < params['latest_rating'])) |
    #         ((df[:, 1] > 0) & (df[:, 2] < params['total_order_amount']))
    # )
    condition_1 = (df[:, 1] != 0) & (df[:, 1] < params['latest_rating'])
    condition_2 = (df[:, 2] != 0) & (df[:, 2] < params['total_order_amount'])
    return jax.numpy.where(condition_1, 1,
                           jax.numpy.where(condition_2, 2, 0))


def processed_monitoring(df, ret_column, months=12):
    """
    模型结果
    """
    data = []
    for index, row in df.iterrows():
        # 添加供应商评分监测
        if ret_column[index] == 1:
            data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": "供应商评分",
                "monitoring_value": row["latest_rating"],
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": row['contact_person'] if 'contact_person' in row else ""
                # "warning_method": rule_df.iloc[0]["warning_method"] if len(rule_df) and "warning_method" in rule_df.columns else "平台消息，短信",
                # "receiver": rule_df.iloc[0]["receiver"] if len(rule_df) and "receiver" in rule_df.columns else ""
            })
        elif ret_column[index] == 2:
            data.append({
                "supplier_name": row["supplier_name"],
                "monitoring_item": f"供应商近{months}个月与核企累计订单金额",
                "monitoring_value": row["latest_rating"],
                "warning_status": True,
                "warning_method": "平台消息，短信",
                "receiver": row['contact_person'] if 'contact_person' in row else ""
                # "warning_method": rule_df.iloc[0]["warning_method"] if len(rule_df) and "warning_method" in rule_df.columns else "平台消息，短信",
                # "receiver": rule_df.iloc[0]["receiver"] if len(rule_df) and "receiver" in rule_df.columns else ""
            })
        # else:
        #     monitoring_data.append({
        #         "supplier_name": row["supplier_name"],
        #         "monitoring_item": "",
        #         "monitoring_value": "",
        #         "warning_status": False,
        #         "warning_method": "",
        #         "receiver": row['contact_person'] if 'contact_person' in row else ""
        #     })
    new_df = pd.DataFrame(data,
                          columns=["supplier_name", "monitoring_item", "monitoring_value", "warning_status",
                                   "warning_method", "receiver"])
    return new_df


def save_ori_file(df, path, feature=None, url=None, payload={}):
    if 'task_id' in payload:
        df['task_id'] = payload['task_id']
    if feature:
        df = df[feature]
    logging.info(f"保存文件 {path}")
    df.to_csv(path, index=False)
    if url:
        logging.info(f"网络请求 {url} ...")
        try:
            import json
            payload["params"] = json.loads(df.to_json(orient="records"))
            logging.info(f"网络请求 {url} ... {payload}")
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                logging.info(f"网络请求 {url} 成功")
            else:
                raise RuntimeError(f"网络请求 {url} 失败, code {response.status_code}")
        except Exception as e:
            logging.error(f"网络请求 {url} 失败, {e}")
            raise RuntimeError(f"网络请求 {url} 失败, {e}")
    return df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
        handlers=[logging.StreamHandler()]  # 将日志输出到终端
    )
    data_endpoint = 'http://10.1.120.42:8080'
    rule_endpoint = 'http://10.1.120.42:8083'

    # df_contract = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=contract")
    # print(df_contract)
    #
    # df_order = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=order")
    # print(df_order)
    #
    # df_receipt = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=warehouse_receipt")
    # print(df_receipt)
    #
    # df_invoice = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=invoice")
    # print(df_invoice)
    #
    # df_arap = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=voucher")
    # print(df_arap)
    #
    # df_payment = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=pay_info")
    # print(df_payment)
    #
    # df_supplier = read_endpoint(f"{data_endpoint}/tmpc/data/list/?type=supplier")
    # print(df_supplier)

    import secretflow as sf

    # sf.shutdown()
    sf.init(parties=['alice', 'bob', 'carol'], address='local')
    alice, bob = sf.PYU('alice'), sf.PYU('bob')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    def marketing():
        # marketing
        data_columns = ['cooperation_duration', 'latest_rating', 'total_order_amount']
        param_columns = ['cooperation_duration', 'latest_rating', 'total_order_amount']
        df_pyu_obj, np_pyu_obj = alice(prepare_data_by_supplier, num_returns=2)(data_endpoint, data_columns)
        params_pyu_obj = bob(prepare_params)(rule_endpoint, param_columns, 'qualified_suppliers')

        from secretflow.device import SPUCompilerNumReturnsPolicy

        np_spu_object = np_pyu_obj.to(spu)
        params_spu_object = params_pyu_obj.to(spu)
        ret_spu_obj = spu(
            process_marketing,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=1,
        )(np_spu_object, params_spu_object)
        logging.info(f"condition_spu_obj: {sf.reveal(ret_spu_obj)}")

        ret_pyu_obj = ret_spu_obj.to(alice)
        result_df = alice(processed_marketing)(df_pyu_obj, ret_pyu_obj)
        logging.info(f"result_df: {sf.reveal(result_df)}")

        alice(save_ori_file)(result_df, "model_marketing.csv")

        # ret_pyu_obj = ret_spu_obj.to(bob)
        # df_pyu_obj = df_pyu_obj.to(bob)
        # np_pyu_obj = np_pyu_obj.to(bob)
        # result_df = bob(processed_marketing)(df_pyu_obj, ret_pyu_obj)
        # logging.info(f"result_df: {sf.reveal(result_df)}")

    def monitoring():
        # monitoring
        data_columns = ['latest_rating', 'total_order_amount']
        param_columns = ['latest_rating', 'total_order_amount']
        df_pyu_obj, np_pyu_obj = alice(prepare_data_by_supplier, num_returns=2)(data_endpoint, data_columns)
        params_pyu_obj = bob(prepare_params)(rule_endpoint, param_columns, 'loan_follow_up')

        from secretflow.device import SPUCompilerNumReturnsPolicy

        np_spu_object = np_pyu_obj.to(spu)
        params_spu_object = params_pyu_obj.to(spu)
        ret_spu_obj = spu(
            process_monitoring,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=1,
        )(np_spu_object, params_spu_object)
        logging.info(f"condition_spu_obj: {sf.reveal(ret_spu_obj)}")

        ret_pyu_obj = ret_spu_obj.to(alice)
        result_df = alice(processed_monitoring)(df_pyu_obj, ret_pyu_obj)
        logging.info(f"result_df: {sf.reveal(result_df)}")

        alice(save_ori_file)(result_df, "model_monitoring.csv")

        # ret_pyu_obj = ret_spu_obj.to(bob)
        # df_pyu_obj = df_pyu_obj.to(bob)
        # np_pyu_obj = np_pyu_obj.to(bob)
        # result_df = bob(processed_monitoring)(df_pyu_obj, ret_pyu_obj)
        # bob_result_df = result_df.to(bob)
        # logging.info(f"result_df: {sf.reveal(result_df)}")


    def available():
        # available
        data_columns = ['cooperation_duration', 'latest_rating', 'avg_order_amount']
        param_columns = ['cooperation_duration', 'latest_rating', 'avg_payment_cycle']
        df_pyu_obj, np_pyu_obj = alice(prepare_data_by_supplier, num_returns=2)(data_endpoint, data_columns)
        params_pyu_obj = bob(prepare_params)(rule_endpoint, param_columns, 'credit_limit')

        from secretflow.device import SPUCompilerNumReturnsPolicy

        np_spu_object = np_pyu_obj.to(spu)
        params_spu_object = params_pyu_obj.to(spu)
        ret_spu_obj = spu(
            process_available,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=1,
        )(np_spu_object, params_spu_object)
        logging.info(f"condition_spu_obj: {sf.reveal(ret_spu_obj)}")

        ret_pyu_obj = ret_spu_obj.to(alice)
        result_df = alice(processed_available)(df_pyu_obj, ret_pyu_obj)
        logging.info(f"result_df: {sf.reveal(result_df)}")

        alice(save_ori_file)(result_df, "model_available.csv")

        # ret_pyu_obj = ret_spu_obj.to(bob)
        # df_pyu_obj = df_pyu_obj.to(bob)
        # np_pyu_obj = np_pyu_obj.to(bob)
        # result_df = bob(processed_monitoring)(df_pyu_obj, ret_pyu_obj)
        # bob_result_df = result_df.to(bob)
        # logging.info(f"result_df: {sf.reveal(result_df)}")


    def withdraw():
        # withdraw
        data_columns = ['credit_amount', 'order_amount_tax_included', 'total_amount_with_tax']
        param_columns = ['financing_balance_param', 'delivered_uninvoiced_amount_param', 'undelivered_amount_param']
        df_pyu_obj, np_pyu_obj = alice(prepare_data_by_order, num_returns=2)(data_endpoint, data_columns)
        params_pyu_obj = bob(prepare_params)(rule_endpoint, param_columns, 'financing_application')

        from secretflow.device import SPUCompilerNumReturnsPolicy

        np_spu_object = np_pyu_obj.to(spu)
        params_spu_object = params_pyu_obj.to(spu)
        ret_spu_obj = spu(
            process_withdraw,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=1,
        )(np_spu_object, params_spu_object)
        logging.info(f"condition_spu_obj: {sf.reveal(ret_spu_obj)}")

        ret_pyu_obj = ret_spu_obj.to(alice)
        result_df = alice(processed_withdraw)(df_pyu_obj, ret_pyu_obj)
        logging.info(f"result_df: {sf.reveal(result_df)}")

        alice(save_ori_file)(result_df, "model_withdraw.csv")

        # ret_pyu_obj = ret_spu_obj.to(bob)
        # df_pyu_obj = df_pyu_obj.to(bob)
        # np_pyu_obj = np_pyu_obj.to(bob)
        # result_df = bob(processed_monitoring)(df_pyu_obj, ret_pyu_obj)
        # bob_result_df = result_df.to(bob)
        # logging.info(f"result_df: {sf.reveal(result_df)}")

    marketing()
    monitoring()
    available()
    withdraw()