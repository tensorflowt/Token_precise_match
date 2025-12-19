import argparse
import dataclasses

from sglang.srt.disaggregation.mini_lb_schedule import PrefillConfig, DecodeConfig


@dataclasses.dataclass
class LBArgs:
    rust_lb: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    policy: str = "random"
    prefill_infos: list = dataclasses.field(default_factory=list)
    decode_infos: list = dataclasses.field(default_factory=list)
    log_interval: int = 5
    timeout: int = 600

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--enable-pd-load-record",
            action="store_true",
            help="enable load-record",
        )
        parser.add_argument(
            "--rust-lb",
            action="store_true",
            help="Deprecated, please use SGLang Router instead, this argument will have no effect.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=LBArgs.host,
            help=f"Host to bind the server (default: {LBArgs.host})",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=LBArgs.port,
            help=f"Port to bind the server (default: {LBArgs.port})",
        )
        parser.add_argument(
            "--policy",
            type=str,
            default=LBArgs.policy,
            choices=["random", "po2"],
            help=f"Policy to use for load balancing (default: {LBArgs.policy})",
        )
        parser.add_argument(
            "--prefill",
            type=str,
            default=[],
            nargs="+",
            help="URLs for prefill servers",
        )
        parser.add_argument(
            "--decode",
            type=str,
            default=[],
            nargs="+",
            help="URLs for decode servers",
        )
        parser.add_argument(
            "--prefill-bootstrap-ports",
            type=int,
            nargs="+",
            help="Bootstrap ports for prefill servers",
        )
        parser.add_argument(
            "--prefill-load-to-lb-ports",
            type=int,
            nargs="+",
            help="Load ports for prefill servers",
        )
        parser.add_argument(
            "--decode-load-to-lb-ports",
            type=int,
            nargs="+",
            help="Load ports for decode servers",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=LBArgs.log_interval,
            help=f"Log interval in seconds (default: {LBArgs.log_interval})",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=LBArgs.timeout,
            help=f"Timeout in seconds (default: {LBArgs.timeout})",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "LBArgs":
        bootstrap_ports = args.prefill_bootstrap_ports
        if bootstrap_ports is None:
            bootstrap_ports = [None] * len(args.prefill)
        elif len(bootstrap_ports) == 1:
            bootstrap_ports = bootstrap_ports * len(args.prefill)
        else:
            if len(bootstrap_ports) != len(args.prefill):
                raise ValueError(
                    "Number of prefill URLs must match number of bootstrap ports"
                )
        if args.enable_pd_load_record:
            prefill_load_to_lb_ports = args.prefill_load_to_lb_ports
            if prefill_load_to_lb_ports is None:
                prefill_load_to_lb_ports = [None] * len(args.prefill)
            elif len(prefill_load_to_lb_ports) == 1:
                prefill_load_to_lb_ports =  prefill_load_to_lb_ports * len(args.prefill)
            else:
                if len(prefill_load_to_lb_ports) != len(args.prefill):
                    raise ValueError(
                        "Number of prefill URLs must match number of prefill_load_to_lb_ports ports"
                    )
            prefill_infos = [
                (url, port, p_lb_port) for url, port, p_lb_port in zip(args.prefill, bootstrap_ports, prefill_load_to_lb_ports)
            ]
            decode_load_to_lb_ports = args.decode_load_to_lb_ports
            if decode_load_to_lb_ports is None:
                decode_load_to_lb_ports = [None] * len(args.decode)
            elif len(decode_load_to_lb_ports) == 1:
                decode_load_to_lb_ports = decode_load_to_lb_ports * len(args.decode)
            else:
                if len(decode_load_to_lb_ports) != len(args.decode):
                    raise ValueError(
                        "Number of decode URLs must match number of decode_load_to_lb_ports ports"
                    )
            #print(decode_load_to_lb_ports)
            #args.decode:['http://127.0.0.1:10003', 'http://127.0.0.1:10010']

            decode_infos = [(url, d_lb_port) for url, d_lb_port in zip(args.decode, decode_load_to_lb_ports)]
            return cls(
                rust_lb=args.rust_lb,
                host=args.host,
                port=args.port,
                policy=args.policy,
                prefill_infos=prefill_infos,
                decode_infos=decode_infos,
                log_interval=args.log_interval,
                timeout=args.timeout,
            )
        else:
            prefill_infos = [
                (url, port) for url, port in zip(args.prefill, bootstrap_ports)
            ]

            return cls(
                rust_lb=args.rust_lb,
                host=args.host,
                port=args.port,
                policy=args.policy,
                prefill_infos=prefill_infos,
                decode_infos=args.decode,
                log_interval=args.log_interval,
                timeout=args.timeout,
            )

    def __post_init__(self):
        if not self.rust_lb:
            assert (
                self.policy == "random"
            ), "Only random policy is supported for Python load balancer"


def main():
    parser = argparse.ArgumentParser(
        description="PD Disaggregation Load Balancer Server"
    )
    LBArgs.add_cli_args(parser)
    args = parser.parse_args()
    lb_args = LBArgs.from_cli_args(args)
    if args.enable_pd_load_record:
        from sglang.srt.disaggregation.mini_lb_schedule import run
        prefill_configs = [PrefillConfig(url, port, load_port) for url, port, load_port in lb_args.prefill_infos]
        decode_configs = [DecodeConfig(url, load_port) for url, load_port in lb_args.decode_infos]
        run(
            prefill_configs,
            decode_configs,
            lb_args.host,
            lb_args.port,
            lb_args.timeout,
        )
    else:
        from sglang.srt.disaggregation.mini_lb import run
        prefill_configs = [PrefillConfig(url, port) for url, port in lb_args.prefill_infos]
        run(
            prefill_configs,
            lb_args.decode_infos,
            lb_args.host,
            lb_args.port,
            lb_args.timeout,
        )

if __name__ == "__main__":
    main()
