import logging
import os

import comet_ml
from omegaconf import DictConfig

from src.utils.io import get_root

logger = logging.getLogger(__name__)


def setup_comet(cfg: DictConfig) -> None:
    if 'comet_ml' not in cfg.training_args.report_to:
        return

    comet_config = get_root() / '.comet.config'
    if comet_config.exists():
        os.environ.setdefault('COMET_CONFIG', str(comet_config))

    os.environ.setdefault('COMET_LOG_ASSETS', 'TRUE' if cfg.writer.log_assets else 'FALSE')
    os.environ.setdefault('COMET_START_ONLINE', 'TRUE' if cfg.writer.mode == 'online' else 'FALSE')

    resolved = comet_ml.get_config()
    if resolved['comet.api_key']:
        logger.info(
            f'Comet: project {resolved["comet.project_name"] or "(default)"}, '
            f'config {os.environ.get("COMET_CONFIG", "not from a file")}'
        )
    else:
        logger.warning(
            'Comet: no api_key found in .comet.config, .env or the environment. '
            'The run will fall back to an offline archive, or fail if Comet cannot '
            'start at all. Set report_to=[] to disable logging entirely.'
        )
