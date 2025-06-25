from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.validators import RegexValidator
from .models import Receipt
from .validators import JapaneseUsernameValidator, validate_japanese_username

class ReceiptForm(forms.ModelForm):
    class Meta:
        model = Receipt
        fields = ['file', 'store_name', 'category', 'logo_confidence', 'text', 'issue_date']
        labels = {
            'file': '領収書画像',
            'store_name': '店舗名',
            'category': 'カテゴリ',
            'logo_confidence': 'ロゴ認識信頼度',
            'text': 'テキスト',
            'issue_date': '発行日',
        }
        widgets = {
            'logo_confidence': forms.NumberInput(attrs={'step': '0.01'}),
            'text': forms.Textarea(attrs={'rows': 3}),
            'issue_date': forms.DateInput(attrs={'type': 'date'}),
        }

class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, required=True, help_text='必須項目です。')
    
    # 日本語対応のユーザー名フィールド
    username = forms.CharField(
        max_length=150,
        validators=[JapaneseUsernameValidator()],
        help_text='150文字以下。英数字、ひらがな、カタカナ、漢字、記号が使用可能です。'
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # フィールドのラベルを日本語化
        self.fields['username'].label = 'ユーザー名'
        self.fields['email'].label = 'メールアドレス'
        self.fields['password1'].label = 'パスワード'
        self.fields['password2'].label = 'パスワード（確認）'
        
        # ヘルプテキストを日本語化（usernameは上で設定済み）
        self.fields['password1'].help_text = '''
        <ul>
            <li>パスワードは8文字以上である必要があります。</li>
            <li>パスワードは数字だけにすることはできません。</li>
            <li>パスワードは一般的すぎるパスワードにすることはできません。</li>
            <li>パスワードはユーザー名と似すぎるパスワードにすることはできません。</li>
        </ul>
        '''
        self.fields['password2'].help_text = '確認のため、同じパスワードを再入力してください。'
    
    def clean_username(self):
        username = self.cleaned_data.get('username')
        # 日本語バリデーション
        username = validate_japanese_username(username)
        # 重複チェック
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('このユーザー名は既に使用されています。')
        return username
