import os
from pathlib import Path
import sys
import glob
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
DEBUG = False
ALLOWED_HOSTS = ['34.146.60.14', 'localhost', '127.0.0.1', 'receiptly.net', 'www.receiptly.net']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # ここに自作アプリも追加
    'receipts',  # receiptsアプリが存在するため追加
    'django.contrib.humanize',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'receipts.middleware.CacheControlMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'receipt_manager.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

print("TEMPLATE DIRS:", [str(d) for d in TEMPLATES[0]['DIRS']], file=sys.stderr)

print("DEBUG: base_site.htmlの候補:", glob.glob(str(BASE_DIR / "templates/admin/base_site.html")), file=sys.stderr)
print("DEBUG: receipts/base.htmlの候補:", glob.glob(str(BASE_DIR / "receipts/templates/base.html")), file=sys.stderr)

WSGI_APPLICATION = 'receipt_manager.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'ja'
TIME_ZONE = 'Asia/Tokyo'
USE_I18N = True
USE_L10N = False
USE_TZ = True

# ログイン関連の設定
LOGIN_REDIRECT_URL = '/dashboard/'
LOGIN_URL = '/accounts/login/'

# ユーザー名のバリデーション設定（日本語対応）
AUTH_USER_MODEL = 'auth.User'

# カスタムバリデーター（日本語ユーザー名を許可）
import re
from django.core.validators import RegexValidator

# 日本語対応のユーザー名バリデーター
JAPANESE_USERNAME_VALIDATOR = RegexValidator(
    regex=r'^[a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF66-\uFF9F\u3000-\u303F\uFF00-\uFFEF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\uFFA0-\uFFEF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\uFFA0-\uFFEF]+$',
    message='ユーザー名には英数字、ひらがな、カタカナ、漢字、記号が使用できます。',
    code='invalid_username'
)

# メール設定（開発環境用）
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
DEFAULT_FROM_EMAIL = 'noreply@receipt-manager.com'

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')  # 本番環境用（collectstatic で集約）
STATICFILES_DIRS = [ os.path.join(BASE_DIR, 'static') ]  # 開発環境用（receipt_manager/static を参照）

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'django.log'),
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'receipts': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# キャッシュ設定（未設定の場合のみ追加）
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# ファイルアップロード関連の設定（settings.pyの末尾に追加）
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024  # 20MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024  # 20MB

# 一時ファイルの保存先
FILE_UPLOAD_TEMP_DIR = '/tmp'
