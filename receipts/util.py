import os
import uuid
import logging
from django.core.files import File
from django.conf import settings
from django.core.files.storage import default_storage
from .models import Receipt
import re
from datetime import datetime

logger = logging.getLogger('django')

# STORE_KEYWORDSをここに移動（views.pyから）
STORE_KEYWORDS = {
    "コンビニ": ["セブン", "ローソン", "ファミマ", "ミニストップ", "デイリーヤマザキ", "サークルK", "サンクス"],
    "スーパー": ["イオン", "イトーヨーカドー", "西友", "ライフ", "マルエツ", "サミット", "コープ", "生協"],
    "ドラッグストア": ["マツモトキヨシ", "サンドラッグ", "ツルハ", "ココカラファイン", "ウエルシア"],
    "飲食店": ["マクドナルド", "ケンタッキー", "スターバックス", "ドトール", "タリーズ"],
    "ガソリンスタンド": ["ENEOS", "出光", "コスモ", "昭和シェル", "JX"],
    "その他": []
}

def extract_store_name(text, store_keywords_dict):
    """テキストから店舗名を抽出"""
    text_upper = text.upper()
    for category, keywords in store_keywords_dict.items():
        for keyword in keywords:
            if keyword.upper() in text_upper:
                return keyword
    return ""

def guess_category_by_store(store_name, store_keywords_dict):
    """店舗名からカテゴリを推測"""
    if not store_name:
        return "その他"
    
    for category, keywords in store_keywords_dict.items():
        if store_name in keywords:
            return category
    return "その他"

def extract_total(text):
    """テキストから合計金額を抽出"""
    # 合計金額のパターンを検索
    patterns = [
        r'合計[：:]\s*(\d{1,3}(?:,\d{3})*)',
        r'合計\s*(\d{1,3}(?:,\d{3})*)',
        r'小計[：:]\s*(\d{1,3}(?:,\d{3})*)',
        r'小計\s*(\d{1,3}(?:,\d{3})*)',
        r'税込[：:]\s*(\d{1,3}(?:,\d{3})*)',
        r'税込\s*(\d{1,3}(?:,\d{3})*)',
        r'総計[：:]\s*(\d{1,3}(?:,\d{3})*)',
        r'総計\s*(\d{1,3}(?:,\d{3})*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            amount_str = match.group(1).replace(',', '')
            if amount_str.isdigit():
                return amount_str
    
    return "0"

def is_receipt_image(img_path):
    """画像がレシートかどうかを判定"""
    # AI判定を一時的にスキップ
    return True

def ocr_google_vision(image_path):
    """Google Vision APIでOCR実行"""
    # 既存のOCR関数をここに移動
    # 実際の実装は既存のviews.pyからコピー
    pass

def process_single_image(request, f, results, job_id=None, total=None, done=None, is_from_zip=False, original_filename=None):
    """単一の画像ファイルを処理して、Receiptオブジェクトを作成する"""
    try:
        filename_for_display = original_filename or f.name
        logger.info(f"ファイル処理開始: {filename_for_display} (サイズ: {f.size} bytes)")
        
        # 拡張子とサイズのチェック
        ext = os.path.splitext(filename_for_display)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.heif']:
            raise ValueError(f"対応していないファイル形式です: {ext}")
        if f.size > 16 * 1024 * 1024:  # 16MB
            raise ValueError("ファイルサイズが16MBを超えています。")

        # 一時保存してAI判定
        temp_name = f"{uuid.uuid4()}_{filename_for_display}"
        temp_path = default_storage.save('tmp/' + temp_name, f)
        temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

        if not is_receipt_image(temp_full_path):
             raise ValueError("この画像はレシート・領収書として認識できませんでした。")

        # OCRはtmp/のファイルで実施
        text = ocr_google_vision(temp_full_path)
        
        # 店舗名、カテゴリ、合計金額を抽出
        store_name = extract_store_name(text, STORE_KEYWORDS)
        category = guess_category_by_store(store_name, STORE_KEYWORDS)
        total_amount_str = extract_total(text)
        total_int = int(total_amount_str) if total_amount_str.isdigit() else 0

        # DB保存
        unique_name = f"{uuid.uuid4()}{ext}"
        receipt = Receipt.objects.create(
            file=File(open(temp_full_path, 'rb'), name=unique_name),
            user=request.user,
            store_name=store_name,
            category=category,
            text=text,
            total_amount=total_int,
        )
        
        # tmpファイル削除
        default_storage.delete(temp_path)
        
        results.append({
            "filename": filename_for_display,
            "status": "OK",
            "message": "アップロード成功",
        })
        return None # エラーなし
    except Exception as e:
        logger.error(f"ファイル処理中にエラーが発生 ({filename_for_display}): {str(e)}", exc_info=True)
        # 一時ファイルが残っていれば削除
        if 'temp_full_path' in locals() and os.path.exists(temp_full_path):
            default_storage.delete(temp_path)
        results.append({
            "filename": filename_for_display,
            "status": "NG",
            "message": f"アップロード失敗: {str(e)}",
        })
        return f"ファイルの処理中にエラーが発生しました ({filename_for_display}): {str(e)}"

# 既存の関数をここに追加
def extract_items(text):
    """テキストから商品アイテムを抽出"""
    # 既存の実装をここにコピー
    pass

def extract_receipt_date(text):
    patterns = [
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
        r'(\d{2})[.](\d{1,2})[.](\d{1,2})',
        r'(\d{4})年(\d{1,2})月(\d{1,2})日'
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                y, mth, d = m.groups()
                if len(y) == 2:
                    y = '20' + y
                return datetime(int(y), int(mth), int(d)).date()
            except Exception:
                continue
    return None
