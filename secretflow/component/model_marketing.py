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

marketing_comp = Component(
    name="marketing",
    domain="user",
    version="0.0.1",
    desc="""marketing model rule calculation""",
)

marketing_comp.str_attr(
    name="task_id",
    desc="task id of the model.",
    is_list=False,
    is_optional=False,
)

marketing_comp.str_attr(
    name="data_endpoint",
    desc="endpoint used to access the data service api.",
    is_list=False,
    is_optional=False,
)

marketing_comp.str_attr(
    name="rule_endpoint",
    desc="endpoint used to access the rule service api.",
    is_list=False,
    is_optional=False,
)

marketing_comp.party_attr(
    name="receiver_parties",
    desc="Party names of receiver for result, all party will be receivers default.",
    list_min_length_inclusive=0,
    list_max_length_inclusive=2,
)

marketing_comp.io(
    io_type=IoType.INPUT,
    name="data_input",
    desc="Individual table for data provider",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="features",
            desc="Column(s) used to output.",
            # col_min_cnt_inclusive=1,
        ),
    ],
)

marketing_comp.io(
    io_type=IoType.INPUT,
    name="rule_input",
    desc="Individual table for rule provider",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="features",
            desc="Column(s) used to output.",
            # col_min_cnt_inclusive=1,
        ),
    ],
)

marketing_comp.io(
    io_type=IoType.OUTPUT,
    name="data_output",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)

marketing_comp.io(
    io_type=IoType.OUTPUT,
    name="rule_output",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)


@marketing_comp.eval_fn
def ss_compare_eval_fn(
        *,
        ctx,
        task_id,
        data_endpoint,
        rule_endpoint,
        receiver_parties,
        data_input,
        data_input_features,
        rule_input,
        rule_input_features,
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

    def read_endpoint(url):
        items = []
        logging.info(f"网络请求 {url} ...")
        try:
            page = 1
            size = 100
            while True:
                response = requests.get(f"{url}&page={page}&size={size}")
                if response.status_code == 200:
                    json_data = response.json()
                    if json_data.get("success"):
                        data = json_data.get("data", [])
                        items.extend(data.get("data", []))
                        if data.get('total_pages', 1) > page:
                            logging.info(f"网络请求 {url} 成功")
                            break
                        page += 1
                    else:
                        raise CompEvalError(f"网络请求 {url} 失败, {json_data.get('message')}")
                else:
                    raise CompEvalError(f"网络请求 {url} 失败, code {response.status_code}")
        except Exception as e:
            raise CompEvalError(f"网络请求 {url} 失败, {e}")
        return pd.DataFrame(items)

    logging.info(f"读取订单数据")
    order_df = wait(
        data_pyu(read_endpoint)(f"{data_endpoint}/tmpc/data/list/?type=order"))
    logging.info(f"读取订单数据成功 {len(order_df)}")

    logging.info(f"读取供应商数据")
    supplier_df = wait(
        data_pyu(read_endpoint)(f"{data_endpoint}/tmpc/data/list/?type=supplier"))
    logging.info(f"读取供应商数据成功 {len(supplier_df)}")

    logging.info(f"读取模型数据")
    model_df = wait(
        rule_pyu(read_endpoint)(f"{rule_endpoint}/tmpc/model/params/?type=qualified_suppliers"))
    logging.info(f"读取模型数据成功 {len(model_df)}")

    def process_order(df, months=12):
        logging.info(f"处理订单数据")

        df["order_date"] = pd.to_datetime(df["order_date"], format="%Y/%m/%d")

        from datetime import datetime
        current_date = pd.Timestamp(datetime.now().strftime("%Y/%m/%d"))
        start_date = current_date - pd.DateOffset(months=months)

        df_recent = df[(df["order_date"] >= start_date) & (df["order_date"] <= current_date)]

        # 按供应商分组计算累计金额
        processed_df = df_recent.groupby("supplier_name")["order_amount_tax_included"].sum().reset_index()
        processed_df.rename(columns={"order_amount_tax_included": f"total_order_amount"}, inplace=True)

        logging.info(f"处理订单数据成功 {len(processed_df)}")
        return processed_df

    def process_model(order_df, supplier_df, model_df):
        logging.info(f"两方处理数据")
        if 'order_date' not in order_df.columns:
            raise CompEvalError("order_date is not in order file")
        if 'order_amount_tax_included' not in order_df.columns:
            raise CompEvalError("order_amount_tax_included is not in order file")
        if 'supplier_name' not in order_df.columns:
            raise CompEvalError("supplier_name is not in order file")

        # if "is_qualified" not in supplier_df.columns:
        #     raise CompEvalError("is_qualified is not in supplier file")
        if 'cooperation_duration' not in supplier_df.columns:
            raise CompEvalError("cooperation_duration is not in supplier file")
        if 'latest_rating' not in supplier_df.columns:
            raise CompEvalError("latest_rating is not in supplier file")

        if 'cooperation_duration' not in model_df.columns:
            raise CompEvalError("cooperation_duration is not in model file")
        if 'latest_rating' not in model_df.columns:
            raise CompEvalError("latest_rating is not in model file")
        if 'total_order_amount' not in model_df.columns:
            raise CompEvalError("total_order_amount is not in model file")

        cooperation_duration = model_df.iloc[0]["cooperation_duration"]
        latest_rating = model_df.iloc[0]["latest_rating"]
        total_order_amount = model_df.iloc[0]["total_order_amount"]
        order_df_processed = process_order(order_df, months=model_df)
        df = supplier_df.merge(order_df_processed, on="supplier_name")
        df["is_qualified"] = df.apply(lambda x: 'true' if (
                x["cooperation_duration"] >= cooperation_duration and x["latest_rating"] >= latest_rating and
                x["total_order_amount"] > total_order_amount) else "false",
                                      axis=1)
        logging.info(f"两方处理数据成功 {len(result_df)}")
        return df

    result_df = spu(process_model)(order_df, supplier_df, model_df)

    def save_ori_file(df, path, features, url):
        df = df[features]
        df.to_csv(path, index=False)
        if url:
            logging.info(f"网络请求 {url} ...")
            try:
                payload = {
                    'task_id': task_id,
                    "params": df.to_json(orient="records")
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    logging.info(f"网络请求 {url} 成功")
                else:
                    raise CompEvalError(f"网络请求 {url} 失败, code {response.status_code}")
            except Exception as e:
                raise CompEvalError(f"网络请求 {url} 失败, {e}")

    if data_party in receiver_parties:
        data_output_csv_filename = os.path.join(ctx.data_dir, f"{data_output}.csv")
        logging.info(f"数据方输出文件 {data_output_csv_filename}")
        wait(data_pyu(save_ori_file)(result_df, data_output_csv_filename, data_input_features,
                                     f'{data_endpoint}/tmpc/model/update/?type=qualified_suppliers'))
        logging.info(f"数据方输出输出文件成功 {data_output_csv_filename}")
    if rule_party in receiver_parties:
        rule_output_csv_filename = os.path.join(ctx.data_dir, f"{rule_output}.csv")
        logging.info(f"规则方输出文件 {rule_output_csv_filename}")
        wait(rule_pyu(save_ori_file)(result_df, rule_output_csv_filename, rule_input_features,
                                     f'{rule_endpoint}/tmpc/model/update/?type=qualified_suppliers'))
        logging.info(f"规则方输出文件成功 {rule_output_csv_filename}")

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
            label_types=[name_types[feature] for feature in data_input_features if feature in name_types],
            labels=[name_types[feature] for feature in data_input_features if feature in name_types],
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
            label_types=[name_types[feature] for feature in data_input_features if feature in name_types],
            labels=[name_types[feature] for feature in data_input_features if feature in name_types],
        )
    )
    rule_output_db.meta.Pack(rule_output_meta)

    return {"data_output": data_output_db, "rule_output": rule_output_db}
