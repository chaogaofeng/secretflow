import os
from secretflow import PYU, SPU
from secretflow.component.component import (
    Component,
    IoType,
)
from secretflow.component.core import download_files
from secretflow.component.gold import *
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
from secretflow.utils import logging

gold_withdraw_comp = Component(
    name="withdraw model",
    domain="gold net",
    version="0.0.1",
    desc="""calculate the actual financable amount according to the order fulfillment situation.""",
)

gold_withdraw_comp.str_attr(
    name="order",
    desc="filter order numbers.",
    is_list=False,
    is_optional=False,
    default_value=[],
)

features = [
    "supplier_name", "core_enterprise_name", "order_number", "order_amount",
    "financing_amount", "application_date", "status", "approved_financing_amount"
]

gold_withdraw_comp.str_attr(
    name="output_data_key",
    desc="column(s) used to output for party data provider.",
    is_list=True,
    is_optional=False,
    default_value=[],
    allowed_values=features
)

gold_withdraw_comp.str_attr(
    name="output_rule_key",
    desc="column(s) used to output for party rule provider.",
    is_list=True,
    is_optional=False,
    default_value=[],
    allowed_values=features,
)

gold_withdraw_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Individual table for party data provider",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)

gold_withdraw_comp.io(
    io_type=IoType.INPUT,
    name="input_rule",
    desc="Individual table for party rule provider",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)

gold_withdraw_comp.io(
    io_type=IoType.OUTPUT,
    name="output_data",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)

gold_withdraw_comp.io(
    io_type=IoType.OUTPUT,
    name="output_rule",
    desc="Output for data",
    types=[DistDataType.INDIVIDUAL_TABLE],
)


@gold_withdraw_comp.eval_fn
def ss_compare_eval_fn(
        *,
        ctx,
        order,
        output_data_key,
        output_rule_key,
        input_data,
        input_rule,
        output_data,
        output_rule,
):
    data_path_info = extract_data_infos(input_data, load_ids=True, load_features=True, load_labels=True)
    data_party = list(data_path_info.keys())[0]
    rule_path_info = extract_data_infos(input_rule, load_ids=True, load_features=True, load_labels=True)
    rule_party = list(rule_path_info.keys())[0]
    logging.info(f"筛选订单列表: {order})")
    logging.info(f"数据参与方: {data_party}")
    logging.info(f"数据方输出文件: {output_data}")
    logging.info(f"数据方输出字段列表: {output_data_key}")
    logging.info(f"规则参与方: {rule_party}")
    logging.info(f"规则方输出文件: {output_rule}")
    logging.info(f"规则方输出字段列表: {output_rule_key}")

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

    data_df = data_pyu(read_file)(input_path[data_party],
                                  ['supplier_name', 'purchaser_name', 'order_number', 'order_amount_tax_included',
                                   'credit_amount', 'total_amount_with_tax'])
    rule_df = rule_pyu(read_file)(input_path[rule_party],
                                  ['financing_balance_param', 'delivered_uninvoiced_amount_param',
                                   'undelivered_amount_param'])

    np_data_pyu_obj, np_column_pyu_obj = data_pyu(prepare_data_by_supplier, num_returns=2)(data_df, order=order,
                                                                                           months=12)
    params_pyu_obj = rule_pyu(prepare_params)(rule_df)

    from secretflow.device import SPUCompilerNumReturnsPolicy

    np_data_spu_object = np_data_pyu_obj.to(spu)
    np_column_spu_obj = np_column_pyu_obj.to(spu)
    params_spu_object = params_pyu_obj.to(spu)
    result_spu_obj = spu(
        process_withdraw,
        num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
        user_specified_num_returns=1,
    )(np_data_spu_object, np_column_spu_obj, params_spu_object)

    result_pyu_obj = result_spu_obj.to(data_pyu)
    result_df = data_pyu(processed_withdraw)(np_data_pyu_obj, np_column_pyu_obj, result_pyu_obj)

    output_data_path = os.path.join(ctx.data_dir, f"{output_data}.csv")
    logging.info(f"数据方输出文件")
    output_data_types = data_pyu(save_file)(output_data_path, result_df, output_data_key)
    logging.info(f"数据方输出输出文件成功")

    output_rule_path = os.path.join(ctx.data_dir, f"{output_rule}.csv")
    logging.info(f"规则方输出文件")
    output_rule_types = data_pyu(save_file)(output_rule_path, result_df, output_rule_key)
    logging.info(f"规则方输出文件成功")

    logging.info("组件输出结果")
    # generate DistData
    output_data_db = DistData(
        name=output_data,
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=output_data_path, party=data_party, format="csv")],
    )
    output_data_meta = IndividualTable(
        schema=TableSchema(
            id_types=output_data_types,
            ids=output_data_key,
        ),
        line_count=-1,
    )
    output_data_db.meta.Pack(output_data_meta)

    output_rule_db = DistData(
        name=output_rule,
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=output_rule_path, party=rule_party, format="csv")],
    )
    output_rule_meta = IndividualTable(
        schema=TableSchema(
            id_types=output_rule_types,
            ids=output_rule_key,
        ),
        line_count=-1,
    )
    output_rule_db.meta.Pack(output_rule_meta)

    return {"output_data": output_data_db, "output_alice": output_rule_db}
