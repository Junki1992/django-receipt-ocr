# Receipt Manager (Receiptly)

領収書のOCR処理と管理を行うDjangoベースのWebアプリケーションです。Google Cloud Vision APIとTesseract OCRを使用して領収書から情報を自動抽出し、ユーザーフレンドリーなインターフェースで管理できます。

## 🚀 主な機能

### 📸 OCR機能
- **Google Cloud Vision API**による高精度な文字認識
- **Tesseract OCR**によるフォールバック処理
- 複数画像形式対応（JPG, PNG, GIF, WebP, HEIC, HEIF）
- 画像の前処理（ノイズ除去、傾き補正、コントラスト強調）
- バッチ処理による複数ファイルの一括アップロード

### 🏪 自動情報抽出
- 店舗名の自動検出
- 合計金額の自動抽出
- 購入日・発行日の自動認識
- 商品項目の詳細抽出
- カテゴリの自動分類（飲食、買い物、移動、交際費など）

### 📊 管理機能
- ダッシュボードによる支出分析
- 月別・カテゴリ別の集計表示
- CSV形式でのデータエクスポート
- 領収書の編集・削除機能
- 一括削除機能

### 👤 ユーザー管理
- ユーザー登録・ログイン機能
- 日本語ユーザー名対応
- パスワード変更機能
- プライバシーポリシー・利用規約

## 🛠️ 技術スタック

### バックエンド
- **Django 4.x** - Webフレームワーク
- **SQLite** - データベース
- **Celery** - 非同期タスク処理
- **Redis** - メッセージブローカー

### OCR・AI
- **Google Cloud Vision API** - 主要OCRエンジン
- **Tesseract OCR** - フォールバックOCR
- **TensorFlow/Keras** - 領収書分類モデル
- **OpenCV** - 画像処理
- **Pillow** - 画像操作

### フロントエンド
- **HTML/CSS/JavaScript** - ユーザーインターフェース
- **Bootstrap** - UIフレームワーク
- **jQuery** - JavaScriptライブラリ

## 📋 必要条件

### システム要件
- Python 3.8以上
- Redis サーバー
- Google Cloud Vision API 認証情報

### 主要なPythonパッケージ
```
Django>=4.0
google-cloud-vision
pytesseract
opencv-python
pillow
tensorflow
celery
redis
python-dotenv
pillow-heif
```

## 🚀 セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd receipt_manager
```

### 2. 仮想環境の作成とアクティベート
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定
`.env`ファイルを作成し、以下の設定を追加：
```env
DJANGO_SECRET_KEY=your-secret-key-here
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

### 5. Google Cloud Vision API認証情報の設定
1. Google Cloud Consoleでプロジェクトを作成
2. Vision APIを有効化
3. サービスアカウントキーを作成
4. 認証情報ファイルを`credentials.json`として保存

### 6. データベースの初期化
```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. スーパーユーザーの作成
```bash
python manage.py createsuperuser
```

### 8. Redisサーバーの起動
```bash
redis-server
```

### 9. Celeryワーカーの起動
```bash
celery -A receipt_manager worker --loglevel=info
```

### 10. 開発サーバーの起動
```bash
python manage.py runserver
```

## 📁 プロジェクト構造

```
receipt_manager/
├── receipt_manager/          # Djangoプロジェクト設定
│   ├── settings.py          # プロジェクト設定
│   ├── urls.py              # メインURL設定
│   └── wsgi.py              # WSGI設定
├── receipts/                # メインアプリケーション
│   ├── models.py            # データモデル
│   ├── views.py             # ビュー関数
│   ├── forms.py             # フォーム
│   ├── urls.py              # URL設定
│   ├── tasks.py             # Celeryタスク
│   └── util.py              # ユーティリティ関数
├── templates/               # HTMLテンプレート
├── static/                  # 静的ファイル
├── media/                   # アップロードファイル
└── manage.py               # Django管理コマンド
```

## 🎯 使用方法

### 1. ユーザー登録・ログイン
- `/signup/` でアカウント作成
- `/accounts/login/` でログイン

### 2. 領収書のアップロード
- ホームページから画像ファイルをアップロード
- 複数ファイルの一括アップロード対応
- リアルタイムでの処理状況表示

### 3. データの確認・編集
- ダッシュボードで全体の支出状況を確認
- 個別の領収書を編集・修正
- 商品項目の詳細確認

### 4. データのエクスポート
- CSV形式でデータをエクスポート
- フィルタリング機能付き

## 🔧 設定

### 開発環境
```python
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
```

### 本番環境
```python
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']
```

### ファイルアップロード設定
- 最大ファイルサイズ: 20MB
- 対応形式: JPG, PNG, GIF, WebP, HEIC, HEIF
- 最大アップロード数: 20ファイル

## 🚀 デプロイ

### Gunicorn + Nginx でのデプロイ
1. Gunicornのインストール
```bash
pip install gunicorn
```

2. 静的ファイルの収集
```bash
python manage.py collectstatic
```

3. Gunicornの起動
```bash
gunicorn receipt_manager.wsgi:application --bind 127.0.0.1:8000
```

4. Nginxの設定例
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location /static/ {
        alias /path/to/your/staticfiles/;
    }
    
    location /media/ {
        alias /path/to/your/media/;
    }
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔍 トラブルシューティング

### よくある問題

1. **OCRが動作しない**
   - Google Cloud Vision API認証情報を確認
   - Tesseractがインストールされているか確認

2. **画像アップロードエラー**
   - ファイルサイズ制限を確認
   - 対応形式かどうか確認

3. **Celeryタスクが実行されない**
   - Redisサーバーが起動しているか確認
   - Celeryワーカーが起動しているか確認

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📞 サポート

問題や質問がある場合は、GitHubのIssuesページでお知らせください。

## 🔄 更新履歴

- **v1.0.0** - 初期リリース
  - 基本的なOCR機能
  - ユーザー管理機能
  - ダッシュボード機能

---

**Receipt Manager** - 領収書管理を簡単に、スマートに。 