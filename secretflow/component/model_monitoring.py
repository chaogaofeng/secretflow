import os
from secretflow import PYU, wait, SPU
from secretflow.component.component import (
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.core import download_files
from secretflow.component.model import *

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

monitoring_comp = Component(
    name="monitoring",
    domain="user",
    version="0.0.1",
    desc="""monitoring model rule calculation""",
)

monitoring_comp.str_attr(
    name="task_id",
    desc="task id of the model.",
    is_list=False,
    is_optional=False,
)

monitoring_comp.str_attr(
    name="supplier",
    desc="suppliers of the model.",
    is_list=True,
    is_optional=True,
    default_value=[]
)

monitoring_comp.str_attr(
    name="data_endpoint",
    desc="endpoint used to access the data service api.",
    is_list=False,
    is_optional=False,
)

monitoring_comp.str_attr(
    name="rule_endpoint",
    desc="endpoint used to access the rule service api.",
    is_list=False,
    is_optional=False,
)

monitoring_comp.party_attr(
    name="receiver_parties",
    desc="Party names of receiver for result, all party will be receivers default.",
    list_min_length_inclusive=0,
    list_max_length_inclusive=2,
)

monitoring_comp.io(
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

monitoring_comp.io(
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

monitoring_comp.io(
    io_type=IoType.OUTPUT,
    name="data_output",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)

monitoring_comp.io(
    io_type=IoType.OUTPUT,
    name="rule_output",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)


@monitoring_comp.eval_fn
def ss_compare_eval_fn(
        *,
        ctx,
        task_id,
        supplier,
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
    if len(receiver_parties) not in (0, 1, 2):
        raise CompEvalError.party_check_failed(
            f"receiver_parties should be empty or have two parties, {receiver_parties}"
        )

    data_path_info = extract_data_infos(data_input, load_ids=True, load_features=True, load_labels=True)
    data_party = list(data_path_info.keys())[0]
    rule_path_info = extract_data_infos(rule_input, load_ids=True, load_features=True, load_labels=True)
    rule_party = list(rule_path_info.keys())[0]
    initiator = ctx.initiator_party if ctx.initiator_party else ""
    logging.info(f"任务发起方: {initiator}")
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

    data_columns = ['latest_rating', 'total_order_amount']
    param_columns = ['latest_rating', 'total_order_amount']
    df_pyu_obj, np_pyu_obj = data_pyu(prepare_data_by_supplier, num_returns=2)(data_endpoint, data_columns, supplier)
    params_pyu_obj = rule_pyu(prepare_params)(rule_endpoint, param_columns, 'loan_follow_up')

    from secretflow.device import SPUCompilerNumReturnsPolicy

    np_spu_object = np_pyu_obj.to(spu)
    params_spu_object = params_pyu_obj.to(spu)
    ret_spu_obj = spu(
        process_monitoring,
        num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
        user_specified_num_returns=1,
    )(np_spu_object, params_spu_object)

    ret_pyu_obj = ret_spu_obj.to(data_pyu)
    ret_df = data_pyu(processed_monitoring)(df_pyu_obj, ret_pyu_obj)

    payload = {
        'task_id': task_id,
        'task_initiator': initiator,
        'task_receiver': receiver_parties,
        'task_receiver_param': [{'node_name': data_party, 'node_recall_param': data_input_feature},
                                {'node_name': rule_party, 'node_recall_param': rule_input_feature}],
        'supplier_name': supplier,
        'order_number': [],
    }
    if data_party in receiver_parties:
        data_output_csv_filename = os.path.join(ctx.data_dir, f"{data_output}.csv")
        logging.info(f"数据方输出文件")
        save_ori_file(ret_df, data_output_csv_filename, data_input_feature,
                      f'{data_endpoint}/tmpc/model/update/?type=credit_limit', payload)
        logging.info(f"数据方输出输出文件成功")

    if rule_party in receiver_parties:
        rule_output_csv_filename = os.path.join(ctx.data_dir, f"{rule_output}.csv")
        logging.info(f"规则方输出文件")
        ret_df = ret_df.to(rule_pyu)
        save_ori_file(ret_df, rule_output_csv_filename, rule_input_feature,
                      f'{rule_endpoint}/tmpc/model/update/?type=credit_limit', payload)
        logging.info(f"规则方输出文件成功")

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
            # DistData.DataRef(
            #     # uri=data_output,
            #     party=data_party,
            #     format="csv",
            # ),
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
            # DistData.DataRef(
            #     # uri=rule_output,
            #     party=rule_party,
            #     format="csv",
            # ),
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
