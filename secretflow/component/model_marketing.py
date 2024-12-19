import os
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

marketing_comp.party_attr(
    name="task_id",
    desc="task id of the marketing model",
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
            name="endpoint",
            desc="endpoint used to access the data service api.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1
        ),
        TableColParam(
            name="features",
            desc="Column(s) used to output.",
            col_min_cnt_inclusive=1,
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
            name="endpoint",
            desc="endpoint used to access the rule service api.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1
        ),
        TableColParam(
            name="features",
            desc="Column(s) used to output.",
            col_min_cnt_inclusive=1,
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
        receiver_parties,
        data_input,
        data_input_endpoint,
        data_input_features,
        rule_input,
        rule_input_endpoint,
        rule_input_features,
        data_output,
        rule_output
):
    if len(receiver_parties) not in (0, 2):
        raise CompEvalError.party_check_failed(
            f"receiver_parties should be empty or have two parties, {receiver_parties}"
        )

    data_path_info = extract_data_infos(data_input)
    data_party = list(data_path_info.keys())[0]
    rule_path_info = extract_data_infos(rule_input)
    rule_party = list(rule_path_info.keys())[0]
    result_party_infos = []
    if data_party in receiver_parties:
        result_party_infos.append(data_path_info)
    if rule_party in receiver_parties:
        result_party_infos.append(rule_path_info)
    print(f"data_party: {data_party}")
    print(f"rule_party: {rule_party}")
    print(f"data_party output: {data_output}")
    print(f"rule_party output: {rule_output}")
    print(f"result_party_infos: {len(result_party_infos) == len(receiver_parties)}")

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
        download_files(ctx, uri, input_path)

    # get spu config from ctx
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    data_pyu = PYU(data_party)
    rule_pyu = PYU(rule_party)

    def read_csv(filepath, endpoint_key, path='mpc/data/list/?type=invoice'):
        import pandas as pd
        import requests
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except:
            df = pd.read_csv(filepath, encoding="gbk")

        data = []
        if endpoint_key not in df.columns:
            raise CompEvalError(f"{endpoint_key} is not in input file")
        else:
            for index, row in df.iterrows():
                endpoint = row[endpoint_key]
                url = f"{endpoint}/{path}"
                print(f"请求url: {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    json_data = response.json()
                    if json_data.get("success"):
                        data.extend(json_data.get("data", []))
                else:
                    raise CompEvalError(f"请求endpoint: {url} 失败")
        return pd.DataFrame(data)

    print(f"读取供应商数据 {input_path[data_party]}")
    supplier_df = data_pyu(read_csv)(filepath=input_path[data_party], endpoint_key=data_input_endpoint, path='mpc/data/list/?type=supplier')
    print(supplier_df)
    print(f"读取订单数据 {input_path[data_party]}")
    order_df = data_pyu(read_csv)(filepath=input_path[data_party], endpoint_key=data_input_endpoint, path='mpc/data/list/?type=order')
    print(order_df)
    print(f"读取模型规则数据 {input_path[rule_party]}")
    model_df = rule_pyu(read_csv)(filepath=input_path[data_party], endpoint_key=rule_input_endpoint, path='tmpc/model/params/?type=qualified_suppliers')
    print(model_df)

    def save_ori_file(df, path, features):
        df = df[features]
        df.to_csv(path, index=False)

    data_output_csv_filename =os.path.join(ctx.data_dir, f"{data_output}.csv")
    print(f"写入data输出文件 {data_output_csv_filename}")
    wait(data_pyu(save_ori_file)(supplier_df, data_output_csv_filename, data_input_features))
    rule_output_csv_filename =os.path.join(ctx.data_dir, f"{rule_output}.csv")
    print(f"写入rule输出文件 {rule_output_csv_filename}")
    wait(rule_pyu(save_ori_file)(model_df, rule_output_csv_filename, rule_input_features))

    imeta = IndividualTable()
    assert data_input.meta.Unpack(imeta)
    name_types = []
    for i, t in zip(list(imeta.schema.ids), list(imeta.schema.id_types)):
        name_types[i] = t

    print("输出结果")
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
