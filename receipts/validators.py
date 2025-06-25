from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
import re

class JapaneseUsernameValidator(RegexValidator):
    """
    日本語対応のユーザー名バリデーター
    英数字、ひらがな、カタカナ、漢字、記号を許可
    """
    regex = r'^[a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF66-\uFF9F\u3000-\u303F\uFF00-\uFFEF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\uFFA0-\uFFEF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\uFFA0-\uFFEF]+$'
    message = 'ユーザー名には英数字、ひらがな、カタカナ、漢字、記号が使用できます。'
    code = 'invalid_username'

def validate_japanese_username(username):
    """
    日本語ユーザー名のバリデーション関数
    """
    if not username:
        raise ValidationError('ユーザー名は必須です。')
    
    if len(username) > 150:
        raise ValidationError('ユーザー名は150文字以下である必要があります。')
    
    # 日本語対応の正規表現でチェック
    pattern = r'^[a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF66-\uFF9F\u3000-\u303F\uFF00-\uFFEF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\uFFA0-\uFFEF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F\u3300-\u33FF\uFE30-\uFE4F\uFF00-\uFFEF\uFFA0-\uFFEF]+$'
    
    if not re.match(pattern, username):
        raise ValidationError('ユーザー名には英数字、ひらがな、カタカナ、漢字、記号が使用できます。')
    
    return username 