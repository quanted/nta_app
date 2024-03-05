import os
import logging
import secrets

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

logger.info("NTA-APP django:settings.py")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_ROOT = os.path.join(PROJECT_ROOT, 'templates/')

DEPLOY_ENV = os.getenv("DEPLOY_ENV", "kube-dev")

SECRET_KEY = os.getenv("SECRET_KEY", "need-to-set-secret-key")

# logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"TEMPLATE_ROOT: {TEMPLATE_ROOT}")
# logger.info(f"DEPLOY_ENV: {DEPLOY_ENV}")

LOGIN_REQUIRED = "true" == os.getenv("LOGIN_REQUIRED", "false").lower()
LOGIN_URL = "/nta/login"
LOGIN_VERBOSE = "true" == os.getenv("LOGIN_VERBOSE", "false").lower()
LOGIN_DURATION = int(os.getenv("LOGIN_DURATION", 86400))
logger.info(f"LOGIN_REQUIRED: {LOGIN_REQUIRED}, LOGIN_URL: {LOGIN_URL}, LOGIN_DURATION: {LOGIN_DURATION}, "
            f"LOGIN_VERBOSE: {LOGIN_VERBOSE}")

if DEPLOY_ENV == "kube-dev":
    DEBUG = True
    CORS_ORIGIN_ALLOW_ALL = True
else:
    DEBUG = False
    CORS_ORIGIN_ALLOW_ALL = False

ALLOWED_HOSTS = ['*']
APPEND_SLASH = True

ADMINS = (
    ('Tom Purucker', 'purucker.tom@epa.gov'),
    ('Jeffrey Minucci', 'minucci.jeffrey@epa.gov'),
    ('Deron Smith', 'smith.deron@epa.gov'),
)

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(TEMPLATE_ROOT),
            os.path.join(TEMPLATE_ROOT, 'drupal_2017'),
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        }
    }
]

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'nta_app',
    'corsheaders'
)

MIDDLEWARE = [
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

if LOGIN_REQUIRED:
    MIDDLEWARE += [
        'login_middleware.RequireLoginMiddleware',
        'login_middleware.Http403Middleware',
        'django.contrib.messages.middleware.MessageMiddleware'
    ]

ROOT_URLCONF = 'urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(PROJECT_ROOT, 'db.sqlite3'),
    },
}

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}

# Internationalization
# https://docs.djangoproject.com/en/1.6/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/New_York'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATICFILES_DIRS = (
    os.path.join(PROJECT_ROOT, 'static'),
)

STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
)

STATIC_ROOT = os.path.join(PROJECT_ROOT, "collected_static/nta")
STATIC_URL = '/nta/static/'

# Define ENVIRONMENTAL VARIABLES
os.environ.update({
    'PROJECT_PATH': PROJECT_ROOT,
    'SITE_SKIN': 'EPA',  # Leave empty ('') for default skin, 'EPA' for EPA skin
    'CONTACT_URL': 'https://www.epa.gov/research/forms/contact-us-about-epa-research',
})

SESSION_EXPIRE_AT_BROWSER_CLOSE = True

WSGI_APPLICATION = 'wsgi.application'
