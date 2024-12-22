import logging
import os
import pandas as pd
import requests

from secretflow import PYU, wait, SPU
from secretflow.component.component import (
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.core import download_files

from secretflow.error_system import CompEvalError

from secretflow.component.data_utils import (
    DistDataType,
    extract_data_infos,
)

from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    StorageConfig,
    TableSchema,
)

withdraw_comp = Component(
    name="withdraw",
    domain="user",
    version="0.0.1",
    desc="""withdraw model rule calculation""",
)

withdraw_comp.str_attr(
    name="task_id",
    desc="task id of the model.",
    is_list=False,
    is_optional=False,
)

withdraw_comp.str_attr(
    name="order_numer",
    desc="order numers of the model.",
    is_list=True,
    is_optional=True,
    default_value=[]
)

withdraw_comp.str_attr(
    name="data_endpoint",
    desc="endpoint used to access the data service api.",
    is_list=False,
    is_optional=False,
)

withdraw_comp.str_attr(
    name="rule_endpoint",
    desc="endpoint used to access the rule service api.",
    is_list=False,
    is_optional=False,
)

withdraw_comp.party_attr(
    name="receiver_parties",
    desc="Party names of receiver for result, all party will be receivers default.",
    list_min_length_inclusive=0,
    list_max_length_inclusive=2,
)

withdraw_comp.io(
    io_type=IoType.INPUT,
    name="data_input",
    desc="Individual table for data provider",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="feature",
            desc="Column(s) used to output.",
            # col_min_cnt_inclusive=1,
        ),
    ],
)

withdraw_comp.io(
    io_type=IoType.INPUT,
    name="rule_input",
    desc="Individual table for rule provider",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="feature",
            desc="Column(s) used to output.",
            # col_min_cnt_inclusive=1,
        ),
    ],
)

withdraw_comp.io(
    io_type=IoType.OUTPUT,
    name="data_output",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)

withdraw_comp.io(
    io_type=IoType.OUTPUT,
    name="rule_output",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)


@withdraw_comp.eval_fn
def ss_compare_eval_fn(
        *,
        ctx,
        task_id,
        order_number,
        data_endpoint,
        rule_endpoint,
        receiver_parties,
        data_input,
        data_input_feature,
        rule_input,
        rule_input_feature,
        data_output,
        rule_output
):
    if len(receiver_parties) not in (0, 2):
        raise CompEvalError.party_check_failed(
            f"receiver_parties should be empty or have two parties, {receiver_parties}"
        )

    data_path_info = extract_data_infos(data_input, load_ids=True, load_features=True, load_labels=True)
    data_party = list(data_path_info.keys())[0]
    rule_path_info = extract_data_infos(rule_input, load_ids=True, load_features=True, load_labels=True)
    rule_party = list(rule_path_info.keys())[0]
    logging.info(f"数据参与方: {data_party}")
    logging.info(f"规则参与方: {rule_party}")
    logging.info(f"输出参与方列表: {receiver_parties})")
    logging.info(f"数据参与方输出: {data_output}")
    logging.info(f"规则参与方输出: {rule_output}")

    input_path = {
        data_party: os.path.join(
            ctx.data_dir, data_path_info[data_party].uri
        ),
        rule_party: os.path.join(ctx.data_dir, rule_path_info[rule_party].uri),
    }
    uri = {
        data_party: data_path_info[data_party].uri,
        rule_party: rule_path_info[rule_party].uri,
    }
    with ctx.tracer.trace_io():
        download_files(ctx, uri, input_path, overwrite=False)

    # get spu config from ctx
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    data_pyu = PYU(data_party)
    rule_pyu = PYU(rule_party)

    def read_data(filepath):
        logging.info(f"读取文件{filepath} ...")
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except:
            df = pd.read_csv(filepath, encoding="gbk")
        logging.info(f"读取文件{filepath} 成功。数量为: {len(df)}")
        return df

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
                        raise CompEvalError(f"网络请求 {url} 失败, {json_data.get('message')}")
                else:
                    raise CompEvalError(f"网络请求 {url} 失败, code {response.status_code}")
        except Exception as e:
            raise CompEvalError(f"网络请求 {url} 失败, {e}")

        df = pd.DataFrame(items)
        if len(df) == 0:
            raise CompEvalError(f"网络请求 {url} 失败, 无数据")

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

        if 'contract_number' not in invoice_df.columns:
            raise RuntimeError("contract_number is not in invoice file")
        if 'total_amount_with_tax' not in invoice_df.columns:
            raise RuntimeError("total_amount_with_tax is not in invoice file")
        invoice_df.rename(columns={"contract_number": "order_number"}, inplace=True)
        invoice_df["total_amount_with_tax"] = pd.to_numeric(invoice_df["total_amount_with_tax"], errors="coerce")

        if 'order_number' not in receipt_df.columns:
            raise RuntimeError("order_number is not in receipt file")
        if 'receipt_amount_tax_included' not in receipt_df.columns:
            raise RuntimeError("receipt_amount_tax_included is not in receipt file")
        receipt_df["receipt_amount_tax_included"] = pd.to_numeric(receipt_df["receipt_amount_tax_included"],
                                                                  errors="coerce")

        if 'contract_number' not in voucher_df.columns:
            raise RuntimeError("contract_number is not in voucher file")
        if 'credit_amount' not in voucher_df.columns:
            raise RuntimeError("credit_amount is not in voucher file")
        voucher_df.rename(columns={"contract_number": "order_number"}, inplace=True)
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
            {"credit_amount": 0, "order_amount_tax_included": 0, "total_amount_with_tax": 0,
             "total_amount_with_tax": 0},
            inplace=True)
        # result_df['approved_financing_amount'] = (result_df['credit_amount'] * 0.9
        #                                           + (result_df['order_amount_tax_included'] - result_df['total_amount_with_tax'])*0.7
        #                                           +(result_df['credit_amount'] - result_df['total_amount_with_tax'])*0.4).round(2)
        data = []
        for _, row in result_df.iterrows():
            approved_financing_amount = round(row['credit_amount'] * 0.9
                                              + (row['order_amount_tax_included'] - row['total_amount_with_tax']) * 0.7
                                              + (row['credit_amount'] - row['total_amount_with_tax']) * 0.4, 2)
            data.append({
                "supplier_name": row["supplier_name"],
                "core_enterprise_name": row['purchaser_name'] if row['purchaser_name'] else "",
                "order_number": row["order_number"],
                "order_amount": row["order_amount_tax_included"],
                "financing_amount": "",
                "application_date": "",
                "status": "",
                "financing_balance": approved_financing_amount,
            })

        result_df = pd.DataFrame(data,
                                 columns=["supplier_name", "core_enterprise_name", "order_number", "order_amount",
                                          "financing_amount", "application_date", "status", "financing_balance"])
        logging.info(f"两方处理数据成功 {len(result_df)}")
        return result_df

    def save_ori_file(df, path, feature, url, task_id):
        if feature:
            df = df[feature]
        logging.info(f"保存文件 {path}")
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
                    raise CompEvalError(f"网络请求 {url} 失败, code {response.status_code}")
            except Exception as e:
                raise CompEvalError(f"网络请求 {url} 失败, {e}")
        return df

    def process_one(task_id, data_endpoint, rule_endpoint, order_number, data_input_feature, rule_input_feature):
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

        if data_party in receiver_parties:
            data_output_csv_filename = os.path.join(ctx.data_dir, f"{data_output}.csv")
            logging.info(f"数据方输出文件")
            save_ori_file(result_df, data_output_csv_filename, data_input_feature,
                                         f'{data_endpoint}/tmpc/model/update/?type=financing_application', task_id)
            logging.info(f"数据方输出输出文件成功")
        if rule_party in receiver_parties:
            rule_output_csv_filename = os.path.join(ctx.data_dir, f"{rule_output}.csv")
            logging.info(f"规则方输出文件")
            save_ori_file(result_df, rule_output_csv_filename, rule_input_feature,
                                         f'{rule_endpoint}/tmpc/model/update/?type=financing_application', task_id)
            logging.info(f"规则方输出文件成功")

        return result_df

    wait(data_pyu(process_one)(task_id, data_endpoint, rule_endpoint, order_number, data_input_feature, rule_input_feature))

    # logging.info(f"读取订单数据")
    # order_df = wait(data_pyu(read_endpoint)(f"{data_endpoint}/tmpc/data/list/?type=order"))
    # logging.info(f"读取订单数据成功")
    #
    # logging.info(f"读取供应商数据")
    # supplier_df = wait(data_pyu(read_endpoint)(f"{data_endpoint}/tmpc/data/list/?type=supplier"))
    # logging.info(f"读取供应商数据成功")
    #
    # logging.info(f"读取模型数据")
    # model_df = wait(rule_pyu(read_endpoint)(f"{rule_endpoint}/tmpc/model/params/?type=qualified_suppliers"))
    # logging.info(f"读取模型数据成功")

    # logging.info(f"读取数据方数据")
    # data_df = wait(data_pyu(read_data)(input_path[data_party]))
    # logging.info(f"读取数据方数据成功")
    #
    # logging.info(f"读取规则方数据")
    # rule_df = wait(rule_pyu(read_data)(input_path[rule_party]))
    # logging.info(f"读取规则方数据成功")

    # logging.info(f"联合处理数据")
    # result_df = spu(process_model)(order_df, supplier_df, model_df, supplier)
    # logging.info(f"联合处理数据成功")

    # if data_party in receiver_parties:
    #     data_output_csv_filename = os.path.join(ctx.data_dir, f"{data_output}.csv")
    #     logging.info(f"数据方输出文件")
    #     data_result_df = result_df.to(data_pyu)
    #     wait(data_pyu(save_ori_file)(data_result_df, data_output_csv_filename, data_input_feature,
    #                                  f'{data_endpoint}/tmpc/model/update/?type=qualified_suppliers', task_id))
    #     logging.info(f"数据方输出输出文件成功")
    # if rule_party in receiver_parties:
    #     rule_output_csv_filename = os.path.join(ctx.data_dir, f"{rule_output}.csv")
    #     logging.info(f"规则方输出文件")
    #     rule_result_df = result_df.to(rule_pyu)
    #     wait(rule_pyu(save_ori_file)(rule_result_df, rule_output_csv_filename, rule_input_feature,
    #                                  f'{rule_endpoint}/tmpc/model/update/?type=qualified_suppliers', task_id))
    #     logging.info(f"规则方输出文件成功")

    imeta = IndividualTable()
    assert data_input.meta.Unpack(imeta)
    name_types = []
    for i, t in zip(list(imeta.schema.ids), list(imeta.schema.id_types)):
        name_types[i] = t

    logging.info("组件输出结果")
    data_output_db = DistData(
        name=data_output,
        type=str(DistDataType.INDIVIDUAL_TABLE),
        system_info=data_input.system_info,
        data_refs=[
            DistData.DataRef(
                uri=data_output,
                party=data_party,
                format="csv",
            ),
        ],
    )
    data_output_meta = IndividualTable(
        schema=TableSchema(
            label_types=[name_types[feature] for feature in data_input_feature if feature in name_types],
            labels=[name_types[feature] for feature in data_input_feature if feature in name_types],
        )
    )
    data_output_db.meta.Pack(data_output_meta)

    rule_output_db = DistData(
        name=rule_output,
        type=str(DistDataType.INDIVIDUAL_TABLE),
        system_info=rule_input.system_info,
        data_refs=[
            DistData.DataRef(
                uri=rule_output,
                party=rule_party,
                format="csv",
            ),
        ],
    )

    rule_output_meta = IndividualTable(
        schema=TableSchema(
            label_types=[name_types[feature] for feature in data_input_feature if feature in name_types],
            labels=[name_types[feature] for feature in data_input_feature if feature in name_types],
        )
    )
    rule_output_db.meta.Pack(rule_output_meta)

    return {"data_output": data_output_db, "rule_output": rule_output_db}
