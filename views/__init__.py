# set up nta_app.views root logger
import logging
import os

DEPLOY_ENV = os.getenv("DEPLOY_ENV", "kube-dev")

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)
logger = logging.getLogger("nta_app.views")
# set up deploy specific logging
if DEPLOY_ENV == "kube-dev":  # log in dev mode
    logger.setLevel(logging.INFO)
    logger.warning("VIEWS - logging in Debug mode!")
else:  # log in production mode
    logger.setLevel(logging.WARNING)
