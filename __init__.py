# set up nta_app root logger
import logging
import os
from datetime import datetime
from pytz import timezone

DEPLOY_ENV = os.getenv("DEPLOY_ENV", "kube-dev")


# obtain datetime object containing the current date/time in UTC-5 (New York timezone)
def get_us_east_timestamp(*args):
    return datetime.now(timezone("US/Eastern")).timetuple()


# Convert logging statement timestamp to the US/Eastern timezone.
logging.Formatter.converter = get_us_east_timestamp

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)
logger = logging.getLogger("nta_app")
# set up deploy specific logging
if DEPLOY_ENV == "kube-dev":  # log in dev mode
    logger.setLevel(logging.INFO)
    logger.warning("Init - logging in Debug mode!")
else:  # log in production mode
    logger.setLevel(logging.WARNING)
