import json

import requests
import os
import dotenv


def query_job(job_id):
    """
    Query job API using the provided job ID.

    :param job_id: The job ID to query
    :return: Response from the API
    """
    if os.path.exists('./kusica.env'):
        dotenv.load_dotenv('./kusica.env', override=True, verbose=True)

    cert_root = os.getenv('KUSCIA_CERT_ROOT')
    endpoint = os.getenv("KUSCIA_API_ENDPOINT")
    # Set file paths
    cert_file = os.path.join(cert_root, "kusciaapi-server.crt")
    key_file = os.path.join(cert_root, "kusciaapi-server.key")
    cacert_file = os.path.join(cert_root, "ca.crt")
    token_file = os.path.join(cert_root, "token")

    # Read the token from the specified file
    def read_token(file_path):
        try:
            with open(file_path, 'r') as token_file:
                return token_file.read().strip()
        except Exception as e:
            raise ValueError(f"Error reading token: {e}")

    # Read the token
    token = read_token(token_file)

    # API endpoint
    url = f"{endpoint}/api/v1/job/query"

    # Payload
    payload = {
        "job_id": job_id
    }

    # Headers
    headers = {
        "Token": token,
        "Content-Type": "application/json"
    }

    # Perform the POST request
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            cert=(cert_file, key_file),
            verify=False
        )
        response.raise_for_status()
        result = response.json()
        status = result['data']['status']
        return status['state']

    except Exception as e:
        raise RuntimeError(f"An error occurred during the API request: {e}")


def create_job(job_id, initiator, max_parallelism, tasks):
    """
    Sends a POST request to create a job.

    Args:
        job_id (str): Unique identifier for the job.
        initiator (str): The initiator of the job.
        max_parallelism (int): Maximum number of parallel tasks.
        tasks (list): List of task configurations.

    Returns:
        dict: Response from the API.
    """
    if os.path.exists('./kusica.env'):
        dotenv.load_dotenv('./kusica.env', override=True, verbose=True)

    cert_root = os.getenv('KUSCIA_CERT_ROOT')
    endpoint = os.getenv("KUSCIA_API_ENDPOINT")
    # Set file paths
    cert_file = os.path.join(cert_root, "kusciaapi-server.crt")
    key_file = os.path.join(cert_root, "kusciaapi-server.key")
    cacert_file = os.path.join(cert_root, "ca.crt")
    token_file = os.path.join(cert_root, "token")

    # Construct certificate paths
    with open(token_file, "r") as f:
        token = f.read().strip()

    url = f"{endpoint}/api/v1/job/create"
    headers = {
        "Token": token,
        "Content-Type": "application/json"
    }

    # Construct payload
    payload = {
        "job_id": job_id,
        "initiator": initiator,
        "max_parallelism": max_parallelism,
        "tasks": tasks
    }

    try:
        # Send request
        response = requests.post(url, headers=headers, json=payload,
                                 cert=(cert_file, key_file),
                                 verify=False
                                 )
        response.raise_for_status()
        result = response.json()
        return result['status']['code'] == 0
    except Exception as e:
        raise RuntimeError(f"An error occurred during the API request: {e}")


def create_marketing_job(job_id, supplier=[]):
    if os.path.exists('./kusica.env'):
        dotenv.load_dotenv('./kusica.env', override=True, verbose=True)

    initiator = os.getenv('KUSCIA_INITIATOR')
    parties = os.getenv('KUSCIA_PARTIES').split(',')
    domains = os.getenv('KUSCIA_MARKET_DOMAINS').split(',')
    oup_parties = os.getenv('KUSCIA_MARKET_PARTIES').split(',')
    data_endpoint = os.getenv('KUSCIA_DATA_ENDPOINT')
    rule_endpoint = os.getenv('KUSCIA_RULE_ENDPOINT')

    task_input_config = {
        "sf_datasource_config": {party: {"id": "default-data-source"} for party in parties},
        "sf_cluster_desc": {
            "parties": parties,
            "devices": [{
                "name": "spu",
                "type": "spu",
                "parties": parties,
                "config": "{\"runtime_config\":{\"protocol\":\"REF2K\",\"field\":\"FM64\"},\"link_desc\":{\"connect_retry_times\":60,\"connect_retry_interval_ms\":1000,\"brpc_channel_protocol\":\"http\",\"brpc_channel_connection_type\":\"pooled\",\"recv_timeout_ms\":1200000,\"http_timeout_ms\":1200000}}"
            }, {
                "name": "heu",
                "type": "heu",
                "parties": parties,
                "config": "{\"mode\": \"PHEU\", \"schema\": \"paillier\", \"key_size\": 2048}"
            }],
            "ray_fed_config": {
                "cross_silo_comm_backend": "brpc_link"
            }
        },
        "sf_node_eval_param": {
            'domain': 'user',
            'name': 'marketing',
            'version': '0.0.1',
            'attr_paths': [
                'input/data_input/feature',
                'input/rule_input/feature',
                'task_id',
                'supplier',
                'data_endpoint',
                'rule_endpoint',
                'receiver_parties'],
            'attrs': [{
                'is_na': True,
                'ss': []
            }, {
                'is_na': True,
                'ss': []
            }, {
                'is_na': False,
                's': job_id
            }, {
                'is_na': True,
                'ss': supplier
            }, {
                'is_na': False,
                's': data_endpoint,
            }, {
                'is_na': False,
                's': rule_endpoint,
            }, {
                'is_na': False,
                'ss': oup_parties
            }]
        },
        'sf_input_ids': domains,
        'sf_output_ids': [f'{job_id}-output-0', f'{job_id}-output-1'],
        'sf_output_uris': [f'{job_id}-output-0.csv', f'{job_id}-output-1.csv']
    }
    tasks = [
        {
            "task_id": job_id,
            "app_image": "secretflow-image",
            "parties": [
                {"domain_id": party} for party in parties
            ],
            "alias": job_id,
            "dependencies": [],
            "task_input_config": json.dumps(task_input_config),
        }
    ]
    print(task_input_config)
    return create_job(job_id, initiator, 2, tasks)


def create_available_job(job_id, supplier=[]):
    if os.path.exists('./kusica.env'):
        dotenv.load_dotenv('./kusica.env', override=True, verbose=True)

    initiator = os.getenv('KUSCIA_INITIATOR')
    parties = os.getenv('KUSCIA_PARTIES').split(',')
    domains = os.getenv('KUSCIA_AVAILABLE_DOMAINS').split(',')
    oup_parties = os.getenv('KUSCIA_AVAILABLE_PARTIES').split(',')
    data_endpoint = os.getenv('KUSCIA_DATA_ENDPOINT')
    rule_endpoint = os.getenv('KUSCIA_RULE_ENDPOINT')

    task_input_config = {
        "sf_datasource_config": {party: {"id": "default-data-source"} for party in parties},
        "sf_cluster_desc": {
            "parties": parties,
            "devices": [{
                "name": "spu",
                "type": "spu",
                "parties": parties,
                "config": "{\"runtime_config\":{\"protocol\":\"REF2K\",\"field\":\"FM64\"},\"link_desc\":{\"connect_retry_times\":60,\"connect_retry_interval_ms\":1000,\"brpc_channel_protocol\":\"http\",\"brpc_channel_connection_type\":\"pooled\",\"recv_timeout_ms\":1200000,\"http_timeout_ms\":1200000}}"
            }, {
                "name": "heu",
                "type": "heu",
                "parties": parties,
                "config": "{\"mode\": \"PHEU\", \"schema\": \"paillier\", \"key_size\": 2048}"
            }],
            "ray_fed_config": {
                "cross_silo_comm_backend": "brpc_link"
            }
        },
        "sf_node_eval_param": {
            'domain': 'user',
            'name': 'available',
            'version': '0.0.1',
            'attr_paths': [
                'input/data_input/feature',
                'input/rule_input/feature',
                'task_id',
                'supplier',
                'data_endpoint',
                'rule_endpoint',
                'receiver_parties'],
            'attrs': [{
                'is_na': True,
                'ss': []
            }, {
                'is_na': True,
                'ss': []
            }, {
                'is_na': False,
                's': job_id
            }, {
                'is_na': True,
                'ss': supplier
            }, {
                'is_na': False,
                's': data_endpoint,
            }, {
                'is_na': False,
                's': rule_endpoint,
            }, {
                'is_na': False,
                'ss': oup_parties
            }]
        },
        'sf_input_ids': domains,
        'sf_output_ids': [f'{job_id}-output-0', f'{job_id}-output-1'],
        'sf_output_uris': [f'{job_id}-output-0.csv', f'{job_id}-output-1.csv']
    }
    tasks = [
        {
            "task_id": job_id,
            "app_image": "secretflow-image",
            "parties": [
                {"domain_id": party} for party in parties
            ],
            "alias": job_id,
            "dependencies": [],
            "task_input_config": json.dumps(task_input_config),
        }
    ]
    print(task_input_config)
    return create_job(job_id, initiator, 2, tasks)


def create_withdraw_job(job_id, order_number=[]):
    if os.path.exists('./kusica.env'):
        dotenv.load_dotenv('./kusica.env', override=True, verbose=True)

    initiator = os.getenv('KUSCIA_INITIATOR')
    parties = os.getenv('KUSCIA_PARTIES').split(',')
    domains = os.getenv('KUSCIA_WITHDRAW_DOMAINS').split(',')
    oup_parties = os.getenv('KUSCIA_WITHDRAW_PARTIES').split(',')
    data_endpoint = os.getenv('KUSCIA_DATA_ENDPOINT')
    rule_endpoint = os.getenv('KUSCIA_RULE_ENDPOINT')

    task_input_config = {
        "sf_datasource_config": {party: {"id": "default-data-source"} for party in parties},
        "sf_cluster_desc": {
            "parties": parties,
            "devices": [{
                "name": "spu",
                "type": "spu",
                "parties": parties,
                "config": "{\"runtime_config\":{\"protocol\":\"REF2K\",\"field\":\"FM64\"},\"link_desc\":{\"connect_retry_times\":60,\"connect_retry_interval_ms\":1000,\"brpc_channel_protocol\":\"http\",\"brpc_channel_connection_type\":\"pooled\",\"recv_timeout_ms\":1200000,\"http_timeout_ms\":1200000}}"
            }, {
                "name": "heu",
                "type": "heu",
                "parties": parties,
                "config": "{\"mode\": \"PHEU\", \"schema\": \"paillier\", \"key_size\": 2048}"
            }],
            "ray_fed_config": {
                "cross_silo_comm_backend": "brpc_link"
            }
        },
        "sf_node_eval_param": {
            'domain': 'user',
            'name': 'withdraw',
            'version': '0.0.1',
            'attr_paths': [
                'input/data_input/feature',
                'input/rule_input/feature',
                'task_id',
                'supplier',
                'data_endpoint',
                'rule_endpoint',
                'receiver_parties'],
            'attrs': [{
                'is_na': True,
                'ss': []
            }, {
                'is_na': True,
                'ss': []
            }, {
                'is_na': False,
                's': job_id
            }, {
                'is_na': True,
                'ss': order_number
            }, {
                'is_na': False,
                's': data_endpoint,
            }, {
                'is_na': False,
                's': rule_endpoint,
            }, {
                'is_na': False,
                'ss': oup_parties
            }]
        },
        'sf_input_ids': domains,
        'sf_output_ids': [f'{job_id}-output-0', f'{job_id}-output-1'],
        'sf_output_uris': [f'{job_id}-output-0.csv', f'{job_id}-output-1.csv']
    }
    tasks = [
        {
            "task_id": job_id,
            "app_image": "secretflow-image",
            "parties": [
                {"domain_id": party} for party in parties
            ],
            "alias": job_id,
            "dependencies": [],
            "task_input_config": json.dumps(task_input_config),
        }
    ]
    print(task_input_config)
    return create_job(job_id, initiator, 2, tasks)


def create_monitoring_job(job_id, supplier=[]):
    if os.path.exists('./kusica.env'):
        dotenv.load_dotenv('./kusica.env', override=True, verbose=True)

    initiator = os.getenv('KUSCIA_INITIATOR')
    parties = os.getenv('KUSCIA_PARTIES').split(',')
    domains = os.getenv('KUSCIA_MONITOR_DOMAINS').split(',')
    oup_parties = os.getenv('KUSCIA_MONITOR_PARTIES').split(',')
    data_endpoint = os.getenv('KUSCIA_DATA_ENDPOINT')
    rule_endpoint = os.getenv('KUSCIA_RULE_ENDPOINT')

    task_input_config = {
        "sf_datasource_config": {party: {"id": "default-data-source"} for party in parties},
        "sf_cluster_desc": {
            "parties": parties,
            "devices": [{
                "name": "spu",
                "type": "spu",
                "parties": parties,
                "config": "{\"runtime_config\":{\"protocol\":\"REF2K\",\"field\":\"FM64\"},\"link_desc\":{\"connect_retry_times\":60,\"connect_retry_interval_ms\":1000,\"brpc_channel_protocol\":\"http\",\"brpc_channel_connection_type\":\"pooled\",\"recv_timeout_ms\":1200000,\"http_timeout_ms\":1200000}}"
            }, {
                "name": "heu",
                "type": "heu",
                "parties": parties,
                "config": "{\"mode\": \"PHEU\", \"schema\": \"paillier\", \"key_size\": 2048}"
            }],
            "ray_fed_config": {
                "cross_silo_comm_backend": "brpc_link"
            }
        },
        "sf_node_eval_param": {
            'domain': 'user',
            'name': 'monitoring',
            'version': '0.0.1',
            'attr_paths': [
                'input/data_input/feature',
                'input/rule_input/feature',
                'task_id',
                'supplier',
                'data_endpoint',
                'rule_endpoint',
                'receiver_parties'],
            'attrs': [{
                'is_na': True,
                'ss': []
            }, {
                'is_na': True,
                'ss': []
            }, {
                'is_na': False,
                's': job_id
            }, {
                'is_na': True,
                'ss': supplier
            }, {
                'is_na': False,
                's': data_endpoint,
            }, {
                'is_na': False,
                's': rule_endpoint,
            }, {
                'is_na': False,
                'ss': oup_parties
            }]
        },
        'sf_input_ids': domains,
        'sf_output_ids': [f'{job_id}-output-0', f'{job_id}-output-1'],
        'sf_output_uris': [f'{job_id}-output-0.csv', f'{job_id}-output-1.csv']
    }
    tasks = [
        {
            "task_id": job_id,
            "app_image": "secretflow-image",
            "parties": [
                {"domain_id": party} for party in parties
            ],
            "alias": job_id,
            "dependencies": [],
            "task_input_config": json.dumps(task_input_config),
        }
    ]
    print(task_input_config)
    return create_job(job_id, initiator, 2, tasks)


if __name__ == '__main__':
    print(create_marketing_job('eccc10'))
    print(query_job('eccc9'))
