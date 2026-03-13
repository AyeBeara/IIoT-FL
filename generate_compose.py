"""Generate a docker-compose YAML with one server and a client for
each machine-type folder under `data/`.

The structure is inspired by Flower's official
`framework/docker/complete/compose.yml` where many client services are
defined.  Instead of hard-coding 33 machines, this script scans the
`data` directory and produces a service definition per subdirectory.

Usage:
    python generate_compose.py [--data-dir PATH] [--output FILE]

The generated file contains a network and the server service; clients
mount `./data` read-only and pass the machine type via environment
and command-line arguments.
"""

import argparse
import os

import yaml
from yaml import SafeDumper


class LiteralStr(str):
    pass


def literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


SafeDumper.add_representer(LiteralStr, literal_representer)


def make_client_service(idx: int, machine_type: str) -> tuple[dict, dict]:
    """Return the service dict for a given machine type."""

    Machine_Type = machine_type
    machine_type = machine_type.lower()

    dockerfile = LiteralStr(
        f"""FROM flwr/superexec:1.26.1

# gcc is required for the fastai quickstart example
USER root
RUN apt-get update \\
    && apt-get -y --no-install-recommends install \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*
USER app

WORKDIR /app
COPY --chown=app:app pyproject.toml README.md LICENSE data/{Machine_Type} ./
RUN sed -i 's/.*flwr\\[simulation\\].*//' pyproject.toml \\
  && python -m pip install -U --no-cache-dir .

ENTRYPOINT [\"flower-superexec\"]
"""
    )

    return {
        "image": "flwr/supernode:1.26.1",
        "command": [
            "--insecure",
            "--superlink",
            "superlink:9092",
            "--clientappio-api-address",
            f"0.0.0.0:{9094+idx}",
            "--isolation",
            "process",
            "--node-config",
            f"machine-type='{machine_type}'",
        ],
        "depends_on": ["superlink"],
    }, {
        "build": {
            "context": ".",
            "dockerfile_inline": dockerfile,
        },
        "command": [
            "--insecure",
            "--plugin-type",
            "clientapp",
            "--appio-api-address",
            f"supernode_{machine_type}:{9094+idx}",
        ],
        "deploy": {"resources": {"limits": {"cpus": "2"}}},
        "stop_signal": "SIGINT",
        "depends_on": [f"supernode_{machine_type}"],
    }


def generate_compose(data_dir: str, output: str) -> None:
    """Scan `data_dir` and write a docker-compose file to `output`."""

    machine_types = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    machine_types.sort()

    services: dict = {
        "superlink": {
            "image": "flwr/superlink:1.26.1",
            "command": ["--insecure", "--isolation", "process"],
            "ports": ["9093:9093"],
        },
        "superexec-serverapp": {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.server",
            },
            "command": [
                "--insecure",
                "--plugin-type",
                "serverapp",
                "--appio-api-address",
                "superlink:9091",
            ],
            "restart": "on-failure:3",
            "depends_on": ["superlink"],
        },
    }

    for idx, machine_type in enumerate(machine_types):
        supernode, superexec = make_client_service(idx, machine_type)
        services[
            f"supernode_{machine_type.lower().replace(' ', '_').replace('-', '_')}"
        ] = supernode
        services[
            f"superexec-clientapp_{machine_type.lower().replace(' ', '_').replace('-', '_')}"
        ] = superexec

    compose = {
        # "networks": {"flwr_network": {"driver": "bridge"}},
        "services": services,
    }

    with open(output, "w") as f:
        yaml.safe_dump(compose, f, default_flow_style=False, sort_keys=False)

    print(f"Generated compose file with {len(machine_types)} clients.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create compose with dynamic clients")
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing machine-type subfolders",
    )
    parser.add_argument(
        "--output",
        default="docker-compose.generated.yml",
        help="Path to write the generated compose YAML",
    )
    args = parser.parse_args()
    generate_compose(args.data_dir, args.output)
