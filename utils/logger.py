import logging
import torch.distributed as dist

logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor.utils").setLevel(logging.CRITICAL)
logging.getLogger("torch._dynamo.symbolic_convert").setLevel(logging.CRITICAL)
logging.getLogger("torch._dynamo.output_graph").setLevel(logging.CRITICAL)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger