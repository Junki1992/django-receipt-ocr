from django.shortcuts import render, get_object_or_404, redirect
from django.core.files.storage import default_storage
import os
from .models import Receipt, ProductItem
from PIL import Image, ImageEnhance
import pytesseract
from google.cloud import vision
import io
from django.contrib.auth.decorators import login_required
import csv
from django.http import HttpResponse, JsonResponse
from django.db.models import Sum, Count
from django.db.models.functions import TruncMonth
from django.contrib.auth import get_user_model
import logging
import json
import cv2
import numpy as np
from django.conf import settings
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from django.utils import timezone
from datetime import timedelta
from collections import defaultdict
import sys
from pytesseract import image_to_string
from pathlib import Path
from django.contrib.admin.views.decorators import staff_member_required
from django.core.paginator import Paginator
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
from .forms import ReceiptForm, SignUpForm  # SignUpFormを追加
from google.api_core.exceptions import GoogleAPICallError
import uuid
from django.core.files import File
# from .tasks import run_ocr_task  # この行をコメントアウトまたは削除
from django.contrib.auth import login
from django.shortcuts import render, redirect
import zipfile
import tempfile
from django.template.loader import render_to_string
from django.core.cache import cache
from .util import (
    STORE_KEYWORDS, extract_store_name, guess_category_by_store, 
    extract_total, is_receipt_image, ocr_google_vision, process_single_image, extract_receipt_date
)
import threading
import time
import shutil
import pillow_heif
import base64

# views.py の先頭で一度だけロード
from tensorflow.keras.models import load_model
MODEL = load_model('receipt_classifier.h5')

# --- OCR精度改善モジュール---
def upscale_if_small(img_cv, min_dim=1000, scale_factor=2):
    h, w = img_cv.shape[:2]
    if max(h, w) < min_dim:
        img_cv = cv2.resize(img_cv, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
    return img_cv


def correct_common_ocr_errors(text):
    replacements = {
        '０': '0', 'Ｏ': '0', 'o': '0', 'O': '0',
        '１': '1', 'Ｉ': '1', 'l': '1', 'i': '1',
        '５': '5', 'Ｓ': '5',
        '８': '8', 'Ｂ': '8',
        '―': '-', 'ー': '-',
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '¥': '￥', '$': '＄'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


def convert_cv2_to_bytes(img_cv):
    is_success, buffer = cv2.imencode(".png", img_cv)
    return buffer.tobytes() if is_success else None


def try_google_ocr(image_bytes, vision_client, max_retries=2):
    from google.cloud import vision
    for attempt in range(max_retries):
        try:
            safe_bytes = ensure_bytes(image_bytes)
            image = vision.Image(content=safe_bytes)
            response = vision_client.document_text_detection(image=image)
            if response.text_annotations:
                return response
        except GoogleAPICallError as e:
            print(f"[OCR RETRY {attempt+1}] Google Vision API failed: {e}")
    return None


def fallback_tesseract(image_cv):
    print("[OCR] Falling back to Tesseract")
    config = '--psm 6 --oem 1 -l jpn'
    return pytesseract.image_to_string(image_cv, config=config)


def run_ocr_with_fallback(img_cv, vision_client):
    img_cv = upscale_if_small(img_cv)
    image_bytes = convert_cv2_to_bytes(img_cv)
    response = try_google_ocr(image_bytes, vision_client)

    if response is None:
        text_result = fallback_tesseract(img_cv)
    else:
        text_result = response.full_text_annotation.text

    text_result = correct_common_ocr_errors(text_result)
    return text_result

# Create your views here.

# Google Cloud Vision APIの認証情報のパスを設定
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(settings.BASE_DIR, 'credentials.json')

ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.heif', '.zip']
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_UPLOAD_FILES = 20  # 例：一度に20枚まで

logger = logging.getLogger('django')

# --- モデルのロード（グローバルで1回だけ） ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'receipt_classifier.h5')
receipt_classifier = load_model(MODEL_PATH)

def preprocess_image(image_path):
    """
    画像の前処理を行う関数
    """
    try:
        # HEIC/HEIFの場合はJPEGに変換
        if image_path.lower().endswith(('.heic', '.heif')):
            heif_file = pillow_heif.read_heif(image_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw"
            )
            # 一時的にJPEGとして保存し直す
            jpeg_path = image_path.rsplit('.', 1)[0] + '.jpg'
            image.save(jpeg_path, format="JPEG")
            image_path = jpeg_path
        # 以降は通常通りPillowで処理
        img = Image.open(image_path)
        
        # グレースケール変換
        img = img.convert('L')
        
        # コントラスト強調
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # PIL画像をOpenCV形式に変換
        img_cv = np.array(img)
        
        # ノイズ除去
        img_cv = cv2.fastNlMeansDenoising(img_cv)
        
        # 二値化
        _, img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- 傾き補正の実装ここから ---
        coords = np.column_stack(np.where(img_cv > 0))
        angle = 0
        if len(coords) > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img_cv.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img_cv = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # --- 傾き補正の実装ここまで ---
        
        # 一時ファイルとして保存
        temp_path = f"{image_path}_processed.jpg"
        cv2.imwrite(temp_path, img_cv)
        
        return temp_path
    except Exception as e:
        logger.error(f"画像の前処理中にエラーが発生: {str(e)}")
        return image_path

def ocr_google_vision_advanced(image_path):
    """
    最新のGoogle Cloud Vision API機能を活用した高度なOCR処理関数
    """
    try:
        logger.info(f"高度なOCR処理開始 - 画像パス: {image_path}")
        
        # 画像の前処理
        processed_image_path = preprocess_image(image_path)
        logger.info(f"前処理後の画像パス: {processed_image_path}")
        
        # 画像の存在確認
        if not os.path.exists(processed_image_path):
            logger.error(f"処理後の画像ファイルが見つかりません: {processed_image_path}")
            return ""
            
        client = vision.ImageAnnotatorClient()
        with io.open(processed_image_path, 'rb') as image_file:
            content = image_file.read()
            logger.info(f"画像ファイル読み込み完了 - サイズ: {len(content)} bytes")
        
        safe_bytes = ensure_bytes(content)
        image = vision.Image(content=safe_bytes)
        
        # 最新のOCR設定を最適化
        logger.info("Google Cloud Vision API（最新機能）にリクエスト送信")
        
        # 1. 基本的なテキスト検出（高精度モード）
        text_response = client.text_detection(
            image=image,
            image_context={
                "language_hints": ["ja", "en"],  # 日本語と英語を優先
                "text_detection_params": {
                    "enable_text_detection_confidence_score": True,
                    "advanced_ocr_options": ["DENSE_TEXT"]  # 密集テキスト対応
                }
            }
        )
        
        # 2. ドキュメントテキスト検出（レイアウト解析）
        document_response = client.document_text_detection(
            image=image,
            image_context={
                "language_hints": ["ja", "en"]
            }
        )
        
        # 3. 手書きテキスト検出（オプション）
        handwritten_response = client.text_detection(
            image=image,
            image_context={
                "language_hints": ["ja", "en"],
                "text_detection_params": {
                    "enable_text_detection_confidence_score": True
                }
            }
        )
        
        # 結果の統合と最適化
        results = []
        confidence_scores = []
        
        # 通常のテキスト検出結果を処理
        if text_response.text_annotations:
            for i, text in enumerate(text_response.text_annotations[1:], 1):  # 最初の要素は全体のテキストなのでスキップ
                confidence = getattr(text, 'confidence', 0)
                if confidence > 0.6:  # 信頼度閾値を少し下げてより多くのテキストを取得
                    results.append(text.description)
                    confidence_scores.append(confidence)
                    logger.info(f"テキスト {i}: 信頼度 {confidence:.2f} - 内容: {text.description[:50]}...")
        
        # ドキュメント検出結果を処理（レイアウト情報付き）
        if document_response.full_text_annotation:
            doc_text = document_response.full_text_annotation.text
            if doc_text.strip():
                results.append(doc_text)
                confidence_scores.append(0.9)  # ドキュメント検出は高信頼度
                logger.info(f"ドキュメント検出結果: {doc_text[:100]}...")
        
        # 手書きテキスト検出結果を処理
        if handwritten_response.text_annotations:
            for i, text in enumerate(handwritten_response.text_annotations[1:], 1):
                confidence = getattr(text, 'confidence', 0)
                if confidence > 0.5:  # 手書きは信頼度閾値を下げる
                    results.append(text.description)
                    confidence_scores.append(confidence)
                    logger.info(f"手書きテキスト {i}: 信頼度 {confidence:.2f} - 内容: {text.description[:50]}...")
        
        # 結果の統合と重複除去
        if results:
            # 重複を除去しつつ、高信頼度の結果を優先
            unique_results = []
            seen_texts = set()
            
            for result, confidence in zip(results, confidence_scores):
                # テキストの正規化（空白、改行の統一）
                normalized = ' '.join(result.split())
                if normalized and len(normalized) > 2 and normalized not in seen_texts:
                    unique_results.append((normalized, confidence))
                    seen_texts.add(normalized)
            
            # 信頼度でソート
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            # 最終結果の生成
            final_result = '\n'.join([text for text, _ in unique_results])
            
            logger.info(f"統合されたOCR結果: {final_result[:200]}...")
            logger.info(f"検出されたテキスト数: {len(unique_results)}")
            logger.info(f"平均信頼度: {sum(conf for _, conf in unique_results) / len(unique_results):.2f}")
            
            # 一時ファイルの削除
            if processed_image_path != image_path:
                try:
                    os.remove(processed_image_path)
                    logger.info("一時ファイルを削除しました")
                except Exception as e:
                    logger.warning(f"一時ファイルの削除に失敗: {str(e)}")
            
            print("=== OCRテキスト ===")
            print(final_result)
            print("==================")
            
            return final_result
        else:
            logger.warning("OCR結果が空です")
            return ""
            
    except Exception as e:
        logger.error(f"高度なOCR処理中にエラーが発生: {str(e)}", exc_info=True)
        return ""

def ocr_google_vision_with_layout_analysis(image_path):
    """
    レイアウト解析機能を活用したOCR処理関数
    """
    try:
        logger.info(f"レイアウト解析OCR処理開始 - 画像パス: {image_path}")
        
        # 画像の前処理
        processed_image_path = preprocess_image(image_path)
        
        client = vision.ImageAnnotatorClient()
        with io.open(processed_image_path, 'rb') as image_file:
            content = image_file.read()
        
        safe_bytes = ensure_bytes(content)
        image = vision.Image(content=safe_bytes)
        
        # ドキュメントテキスト検出（レイアウト解析機能）
        response = client.document_text_detection(
            image=image,
            image_context={
                "language_hints": ["ja", "en"]
            }
        )
        
        if response.full_text_annotation:
            # レイアウト情報を活用したテキスト抽出
            layout_text = extract_text_with_layout(response.full_text_annotation)
            logger.info(f"レイアウト解析結果: {layout_text[:200]}...")
            
            # 一時ファイルの削除
            if processed_image_path != image_path:
                try:
                    os.remove(processed_image_path)
                except Exception as e:
                    logger.warning(f"一時ファイルの削除に失敗: {str(e)}")
            
            return layout_text
        else:
            logger.warning("レイアウト解析結果が空です")
            return ""
            
    except Exception as e:
        logger.error(f"レイアウト解析OCR処理中にエラーが発生: {str(e)}", exc_info=True)
        return ""

def extract_text_with_layout(full_text_annotation):
    """
    レイアウト情報を活用してテキストを抽出する関数
    """
    try:
        # ページごとの処理
        pages = full_text_annotation.pages
        extracted_text = []
        
        for page in pages:
            # ブロックごとの処理
            for block in page.blocks:
                block_text = []
                
                # 段落ごとの処理
                for paragraph in block.paragraphs:
                    paragraph_text = []
                    
                    # 単語ごとの処理
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        paragraph_text.append(word_text)
                    
                    # 段落のテキストを結合
                    if paragraph_text:
                        block_text.append(' '.join(paragraph_text))
                
                # ブロックのテキストを結合
                if block_text:
                    extracted_text.append('\n'.join(block_text))
        
        return '\n\n'.join(extracted_text)
        
    except Exception as e:
        logger.error(f"レイアウト情報抽出中にエラーが発生: {str(e)}")
        return ""

# 既存の関数を新しい関数で置き換える（オプション）
def ocr_google_vision(image_path):
    """
    既存の関数を新しい高度なOCR関数で置き換え
    """
    return ocr_google_vision_advanced(image_path)

def guess_item_category(item_name):
    """商品名からカテゴリを推測する"""
    categories = {
        '飲食': ['コーヒー', '紅茶', 'ジュース', '水', 'お茶', 'ビール', 'ワイン', '酒',
                'サンドイッチ', 'パン', 'ケーキ', 'デザート', 'アイス', 'おにぎり',
                '弁当', 'ランチ', '定食', '麺', 'パスタ', 'ピザ', 'ハンバーガー',
                'フライドチキン', '寿司', '刺身', '焼肉', '鍋', 'カレー', '丼',
                'タコス', 'セット', 'メキシコ', 'チリ', 'サルサ', 'ワカモレ',
                'ナチョス', 'ブリトー', 'ファヒータ', 'ケサディーヤ'],
        '日用品': ['洗剤', '石鹸', 'シャンプー', 'リンス', '歯磨き', 'トイレットペーパー',
                  'ティッシュ', 'マスク', 'ハンカチ', 'タオル', '下着', '靴下'],
        '食品': ['米', 'パスタ', '麺', '調味料', '油', '砂糖', '塩', '醤油', '味噌',
                '缶詰', 'レトルト', '冷凍食品', '野菜', '果物', '肉', '魚', '卵',
                '牛乳', 'チーズ', 'ヨーグルト', 'スナック', 'チップス', 'ポップコーン'],
        '文具': ['ノート', 'ペン', '鉛筆', '消しゴム', 'ファイル', 'ホッチキス',
                'クリップ', '付箋', 'マーカー'],
        'その他': []  # デフォルトカテゴリ
    }
    item_lower = item_name.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in item_lower:
                return category
    return 'その他'

# 例: 「1 シロップ レモン 600ml瓶 1 本 ※ 417 417」
item_pattern = re.compile(
    r'([0-9]+)?\s*([一-龥ぁ-んァ-ンA-Za-z0-9\-・\(\)]+)\s+([0-9]+)\s*本.*?([0-9,]{2,7})\s+([0-9,]{2,7})'
)

def extract_items(text):
    """OCRテキストから商品明細を抽出する関数（現場パターン対応版）"""
    import re
    items = []
    
    # 除外ワードリスト（最小限に）
    exclude_names = [
        '合計', '小計', 'お預り', 'お釣り', '点数', '明細', '領収証', 
        '税込', '税額', '合計金額', '差引合計', '商品コード', '商品名', '単価', '額',
        'SHINJO', 'TEL', 'FAX', '金', '備考', '受領印', '消費税額', '税込金額'
    ]
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        # パターン1: 「商品名 数量P ¥金額」の形式（ドン・キホーテなど）
        # 例: 「タコスセット 10P ¥1,780」
        pattern1 = re.compile(r'([一-龥ぁ-んァ-ンA-Za-z0-9\-・\(\)]+)\s+([0-9]+)P\s*¥([0-9,]+)')
        
        # パターン2: 「商品名 数量 本 ※ 単価 金額」の形式
        # 例: 「シロップ レモン 600ml瓶 1 本 ※ 417 417」
        pattern2 = re.compile(r'([一-龥ぁ-んァ-ンA-Za-z0-9\-・\(\)]+)\s+([0-9]+)\s*本\s*※\s*([0-9,]+)\s+([0-9,]+)')
        
        # パターン3: 「商品名 数量本 ※ 単価 金額」の形式（本の前の空白なし）
        # 例: 「サッポロ沖縄パイナップルシロップ 500mlパック 1本 ※ 594 594」
        pattern3 = re.compile(r'([一-龥ぁ-んァ-ンA-Za-z0-9\-・\(\)]+)\s+([0-9]+)本\s*※\s*([0-9,]+)\s+([0-9,]+)')
        
        # パターン4: 「商品名 金額」の形式（従来パターン）
        pattern4 = re.compile(r'([一-龥ぁ-んァ-ンA-Za-z0-9\-・\(\)]+)\s+([0-9,]{2,7})')
        
        # パターン1で試行（ドン・キホーテ形式）
        match = pattern1.search(line)
        if match:
            name = match.group(1).strip()
            quantity = match.group(2)
            price = match.group(3).replace(',', '')
            
            # 除外ワードチェック
            if any(ex in name for ex in exclude_names):
                continue
                
            if name and price.isdigit():
                price_int = int(price)
                if 10 <= price_int <= 100000:  # 10円〜10万円の範囲
                    items.append({
                        'name': name,
                        'price': price_int,
                        'category': guess_item_category(name)
                    })
                    continue  # 次の行へ
        
        # パターン2で試行
        match = pattern2.search(line)
        if match:
            name = match.group(1).strip()
            quantity = match.group(2)
            unit_price = match.group(3).replace(',', '')
            total_price = match.group(4).replace(',', '')
            
            # 除外ワードチェック
            if any(ex in name for ex in exclude_names):
                continue
                
            if name and total_price.isdigit():
                price = int(total_price)
                if 10 <= price <= 100000:  # 10円〜10万円の範囲
                    items.append({
                        'name': name,
                        'price': price,
                        'category': guess_item_category(name)
                    })
                    continue  # 次の行へ
        
        # パターン3で試行
        match = pattern3.search(line)
        if match:
            name = match.group(1).strip()
            quantity = match.group(2)
            unit_price = match.group(3).replace(',', '')
            total_price = match.group(4).replace(',', '')
            
            # 除外ワードチェック
            if any(ex in name for ex in exclude_names):
                continue
                
            if name and total_price.isdigit():
                price = int(total_price)
                if 10 <= price <= 100000:  # 10円〜10万円の範囲
                    items.append({
                        'name': name,
                        'price': price,
                        'category': guess_item_category(name)
                    })
                    continue  # 次の行へ
        
        # パターン4で試行（従来パターン）
        match = pattern4.search(line)
        if match:
            name = match.group(1).strip()
            price = match.group(2).replace(',', '')
            
            # 除外ワードチェック
            if any(ex in name for ex in exclude_names):
                continue
                
            if name and price.isdigit():
                price_int = int(price)
                if 10 <= price_int <= 100000:  # 10円〜10万円の範囲
                    items.append({
                        'name': name,
                        'price': price_int,
                        'category': guess_item_category(name)
                    })
    
    # 絶対に「抽出失敗」は返さない
    if not items:
        # 合計金額から推測して1つの商品として返す
        total = extract_total(text)
        if total and total != "1":
            items.append({
                'name': '商品明細',
                'price': int(total),
                'category': 'その他'
            })
        else:
            # 最後の手段：テキストから数字を探して返す
            numbers = re.findall(r'([0-9,]{3,7})', text)
            if numbers:
                valid_prices = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit() and 100 <= int(n.replace(',', '')) <= 100000]
                if valid_prices:
                    items.append({
                        'name': '商品明細',
                        'price': max(valid_prices),
                        'category': 'その他'
                    })
    
    return items

def extract_total(text):
    """OCRテキストから合計金額を抽出する関数（現場パターン対応版）"""
    import re
    
    # パターン1: 「差引合計 ¥金額」の形式（最優先）
    match = re.search(r'差引合計[^\d]*([0-9,]+)', text)
    if match:
        amount = int(match.group(1).replace(',', ''))
        return str(amount)
    
    # パターン2: 「¥金額-」の形式（領収証の合計金額）
    match = re.search(r'¥([0-9,]+)-', text)
    if match:
        amount = int(match.group(1).replace(',', ''))
        return str(amount)
    
    # パターン3: 「合計 ¥金額」の形式
    match = re.search(r'合計\s*¥([0-9,]+)', text)
    if match:
        amount = int(match.group(1).replace(',', ''))
        return str(amount)
    
    # パターン4: 「お買上点数」の前にある金額（合計金額の可能性が高い）
    match = re.search(r'([0-9,]{3,7})\s*お買上点数', text)
    if match:
        amount = int(match.group(1).replace(',', ''))
        return str(amount)
    
    # パターン5: 「小計」と「税額」から合計を計算
    subtotal_match = re.search(r'小計\s*¥([0-9,]+)', text)
    tax_match = re.search(r'税額\s*¥([0-9,]+)', text)
    if subtotal_match and tax_match:
        subtotal = int(subtotal_match.group(1).replace(',', ''))
        tax = int(tax_match.group(1).replace(',', ''))
        total = subtotal + tax
        return str(total)
    
    # パターン6: 「¥金額」の最大値を返す（お釣りや預かり金は除外）
    exclude_patterns = [
        r'お釣り\s*¥([0-9,]+)',
        r'お預り\s*¥([0-9,]+)',
        r'預かり\s*¥([0-9,]+)',
        r'チNo.*?¥([0-9,]+)',
        r'代として\s*¥([0-9,]+)'
    ]
    exclude_amounts = set()
    for pattern in exclude_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            exclude_amounts.add(int(match.replace(',', '')))
    
    matches = re.findall(r'¥([0-9,]+)', text)
    valid_amounts = []
    for match in matches:
        amount = int(match.replace(',', ''))
        if amount not in exclude_amounts and 10 <= amount <= 100000:
            valid_amounts.append(amount)
    
    if valid_amounts:
        return str(max(valid_amounts))
    
    # パターン7: 全体から最大値を取得（最終手段）
    all_amounts = []
    numbers = re.findall(r'([0-9,]{3,7})', text)
    for num in numbers:
        clean_num = num.replace(',', '')
        if clean_num.isdigit():
            amount = int(clean_num)
            if 100 <= amount <= 100000:
                all_amounts.append(amount)
    if all_amounts:
        return str(max(all_amounts))
    
    # 最終手段：1円でも返す
    return ""

def extract_store_name(text, store_keywords_dict):
    import re
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return '不明'
    scan_lines = lines[:8] + lines[-20:]
    exclude_store_names = ['PASMO', 'Suica', 'ICOCA', 'SUGOCA', 'manaca', 'TOICA', 'Kitaca', 'nimoca', 'はやかけん']
    for line in scan_lines:
        if any(ex in line for ex in exclude_store_names):
            continue
        for group, info in store_keywords_dict.items():
            for keyword in info['keywords']:
                if keyword.lower() in line.lower():
                    return keyword
    for line in scan_lines:
        if any(ex in line for ex in exclude_store_names):
            continue
        # 「株式会社」から始まる
        m = re.match(r'(株式会社[^\s]*)', line)
        if m:
            return m.group(1)
        # 「店」「商店」などで終わる
        m = re.search(r'([^\s]+(店|商店|カフェ|支店|会社))', line)
        if m:
            return m.group(1)
        if re.search(r'(株式会社|商店|店|支店|カフェ|会社)', line):
            return line
    return '不明'

def normalize_store_name_for_category(name):
    import re
    name = name.lower()
    name = re.sub(r'\s+', '', name)  # 空白除去
    name = re.sub(r'[^\w\-ー一-龥ぁ-んァ-ン]', '', name)  # 記号除去
    # セブンイレブン系
    name = name.replace('7セブンイレブン', 'セブンイレブン')
    name = name.replace('7eleven', 'セブンイレブン')
    name = name.replace('seveneleven', 'セブンイレブン')
    name = name.replace('セブン-イレブン', 'セブンイレブン')
    name = name.replace('セブン‐イレブン', 'セブンイレブン')
    name = name.replace('seven', 'セブン')
    name = name.replace('7', 'セブン')
    # LAWSON
    name = name.replace('lawson', 'ローソン')
    return name

def guess_category_by_store(store_name, store_keywords_dict):
    norm_name = normalize_store_name_for_category(store_name)
    print(f"[DEBUG] 店舗名: {store_name} → 正規化: {norm_name}")
    for group, info in store_keywords_dict.items():
        for keyword in info['keywords']:
            norm_kw = normalize_store_name_for_category(keyword)
            print(f"[DEBUG] 比較: '{norm_kw}' in '{norm_name}'")
            if norm_kw in norm_name:
                print(f"[DEBUG] カテゴリヒット: {info['category']} ({norm_kw} in {norm_name})")
                return info['category']
    print("[DEBUG] カテゴリ: その他")
    return "その他"

def is_receipt_image(img_path):
    # AI判定を一時的にスキップ
    return True

def crop_top_region(pil_image, ratio=0.2):
    img = np.array(pil_image)
    h, w = img.shape[:2]
    cropped = img[:int(h * ratio), :]
    return Image.fromarray(cropped)

def extract_store_name_from_image(image):
    logo_area = crop_top_region(image)
    text = image_to_string(logo_area, lang='jpn')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    candidates = [line for line in lines if re.search(r'(株式会社|商店|店|支店|カフェ|会社)', line)]
    return candidates[0] if candidates else None

def detect_logo_and_category(image_path):
    """
    Google Cloud Vision APIのロゴ検出機能を利用して、
    画像からロゴを検出し、店舗名・カテゴリを判別する関数
    """
    try:
        # Vision APIクライアントの初期化
        client = vision.ImageAnnotatorClient()

        # 画像を読み込み
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        safe_bytes = ensure_bytes(content)
        image = vision.Image(content=safe_bytes)

        # ロゴ検出を実行
        response = client.logo_detection(image=image)
        logos = response.logo_annotations

        # 店舗名・カテゴリ判定
        detected_store = None
        category = "その他"
        best_score = 0.0
        for logo in logos:
            desc = logo.description.lower()
            score = logo.score
            if score < 0.7:
                continue  # 信頼度が低いものは無視
            if "mcdonald" in desc or "マクドナルド" in desc:
                detected_store = "マクドナルド"
                category = "飲食"
                best_score = score
                break
            elif "seven" in desc or "セブン" in desc or "7-11" in desc or "7-eleven" in desc:
                detected_store = "セブンイレブン"
                category = "買い物"
                best_score = score
                break
            # ...他の店舗も同様に...
        if not detected_store:
            store_name = extract_store_name_from_image(Image.open(image_path))
            # STORE_KEYWORDSを渡す
            category = guess_category_by_store(store_name, STORE_KEYWORDS) if store_name else "その他"
            detected_store = store_name or "不明"
        return {
            'store_name': detected_store,
            'category': category,
            'confidence': best_score
        }

    except Exception as e:
        logger.error(f"ロゴ検出中にエラーが発生: {str(e)}", exc_info=True)
        return {
            'store_name': "不明",
            'category': "その他",
            'confidence': 0.0
        }

def process_single_image(request, f, results, job_id=None, total=None, done=None, is_from_zip=False, original_filename=None):
    """単一の画像ファイルを処理して、Receiptオブジェクトを作成する"""
    try:
        filename_for_display = original_filename or f.name
        if not filename_for_display or filename_for_display in ['image.heic', 'image.heif', 'image.jpg', 'image.jpeg']:
            import time
            ext = os.path.splitext(f.name)[1].lower() or '.jpg'
            filename_for_display = f"photo_{int(time.time())}{ext}"
        logger.info(f"ファイル処理開始: {filename_for_display} (サイズ: {f.size} bytes)")

        # 拡張子とサイズのチェック
        ext = os.path.splitext(filename_for_display)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"対応していないファイル形式です: {ext}")
        if f.size > MAX_FILE_SIZE:
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
        total_int = int(total_amount_str) if total_amount_str.isdigit() else None

        # 発行日を抽出
        issue_date = extract_receipt_date(text)

        # DB保存
        unique_name = f"receipts/{uuid.uuid4()}{ext}"
        receipt = Receipt.objects.create(
            file=File(open(temp_full_path, 'rb'), name=unique_name),
            user=request.user,
            store_name=store_name,
            category=category,
            text=text,
            total_amount=total_int,
            issue_date=issue_date,
        )
        # run_ocr_task.delay(receipt.id)  # この行をコメントアウト

        # tmpファイル削除
        default_storage.delete(temp_path)

        results.append({
            "filename": filename_for_display,
            "status": "OK",
            "message": "アップロード成功",
        })
        print(f"[DEBUG] キャッシュ保存 job_id={job_id}, results={results}")
        cache.set(f'results_{job_id}', results, timeout=3600)
        cache.set(f'error_{job_id}', None, timeout=3600)
        return None # エラーなし
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"ファイル処理中にエラーが発生 ({filename_for_display}): {str(e)}", exc_info=True)
        # 一時ファイルが残っていれば削除
        if 'temp_full_path' in locals() and os.path.exists(temp_full_path):
            if 'temp_path' in locals():
                default_storage.delete(temp_path)
        results.append({
            "filename": filename_for_display,
            "status": "NG",
            "message": f"アップロード失敗: {str(e)}",
        })
        print(f"[DEBUG] キャッシュ保存 job_id={job_id}, results={results}")
        cache.set(f'results_{job_id}', results, timeout=3600)
        cache.set(f'error_{job_id}', str(e), timeout=3600)
        return f"ファイルの処理中にエラーが発生しました ({filename_for_display}): {str(e)}"

@login_required
def index(request):
    print("=== indexビュー呼び出し ===")
    print("リクエストメソッド:", request.method)
    print("FILES内容:", request.FILES)
    error = None
    results = []

    if request.method == "POST":
        if not request.FILES.get('receipt'):
            return JsonResponse({'success': False, 'error': 'ファイルが選択されていません。もう一度やり直してください。'})
        try:
            job_id = request.POST.get('job_id')
            files = request.FILES.getlist("receipt")
            total = 0
            done = 0

            # --- 合計画像枚数カウント ---
            for f in files:
                ext = os.path.splitext(f.name)[1].lower()
                if ext == '.zip':
                    # zipファイルを一時保存
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_zip:
                        for chunk in f.chunks():
                            tmp_zip.write(chunk)
                        tmp_zip_path = tmp_zip.name
                    try:
                        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                            for file_info in zip_ref.infolist():
                                if not file_info.is_dir() and not file_info.filename.startswith('__MACOSX') and not file_info.filename.startswith('._'):
                                    zip_ext = os.path.splitext(file_info.filename)[1].lower()
                                    if zip_ext in ALLOWED_EXTENSIONS:
                                        total += 1
                    except Exception as e:
                        print(f"DEBUG: zipファイル枚数カウントエラー: {e}")
                    finally:
                        os.remove(tmp_zip_path)
                elif ext in ALLOWED_EXTENSIONS:
                    total += 1

            set_progress(job_id, total, 0)

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                saved_files = []
                for f in files:
                    temp_name = f"{uuid.uuid4()}_{f.name}"
                    temp_path = default_storage.save('tmp/' + temp_name, f)
                    saved_files.append({
                        'path': temp_path,
                        'name': f.name,
                        'size': f.size
                    })

                def process_files():
                    nonlocal done, error, results
                    for file_info in saved_files:
                        temp_path = file_info['path']
                        original_name = file_info['name']
                        temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
                        ext = os.path.splitext(original_name)[1].lower()
                        if ext == '.zip':
                            # zip展開・画像処理
                            try:
                                with zipfile.ZipFile(temp_full_path, 'r') as zip_ref:
                                    # 一時ディレクトリを作成
                                    extract_dir = os.path.join(settings.MEDIA_ROOT, 'tmp', f'extract_{uuid.uuid4()}')
                                    os.makedirs(extract_dir, exist_ok=True)
                                    
                                    # zipファイルを展開
                                    zip_ref.extractall(extract_dir)
                                    
                                    # 展開されたファイルを処理
                                    for root, dirs, files in os.walk(extract_dir):
                                        for file in files:
                                            if not file.startswith('__MACOSX') and not file.startswith('._'):
                                                file_path = os.path.join(root, file)
                                                file_ext = os.path.splitext(file)[1].lower()
                                                if file_ext in ALLOWED_EXTENSIONS:
                                                    # ファイル名の文字化け修正
                                                    fixed_name = file
                                                    try:
                                                        fixed_name = file.encode('cp437').decode('utf-8')
                                                    except Exception:
                                                        pass  # デコード失敗時はそのまま
                                                    # 1. 一時ファイルをDjangoストレージに保存
                                                    with open(file_path, 'rb') as img_file:
                                                        temp_name = f"{uuid.uuid4()}_{fixed_name}"
                                                        temp_path = default_storage.save('tmp/' + temp_name, img_file)
                                                    # 2. そのパスを使ってprocess_single_imageを呼ぶ
                                                    temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
                                                    with open(temp_full_path, 'rb') as f:
                                                        django_file = File(f, name=fixed_name)
                                                        processing_error = process_single_image(
                                                            request, django_file, results,
                                                            job_id=job_id, total=total, done=done,
                                                            is_from_zip=True, original_filename=fixed_name
                                                        )
                                                        if processing_error:
                                                            error = processing_error
                                                            break
                                                    # 3. 進捗を更新
                                                    done += 1
                                                    set_progress(job_id, total, done)
                                                    time.sleep(0.2)
                                                    # 4. tmpファイル削除
                                                    default_storage.delete(temp_path)
                            except Exception as e:
                                print(f"DEBUG: zipファイル処理エラー: {e}")
                                error = f"zipファイルの処理中にエラーが発生しました: {str(e)}"
                        elif ext in ALLOWED_EXTENSIONS:
                            done += 1
                            set_progress(job_id, total, done)
                            time.sleep(0.2)
                            with open(temp_full_path, 'rb') as f:
                                django_file = File(f, name=original_name)
                                processing_error = process_single_image(
                                    request, django_file, results,
                                    job_id=job_id, total=total, done=done
                                )
                                if processing_error:
                                    error = processing_error
                            try:
                                default_storage.delete(temp_path)
                            except Exception as e:
                                print(f"DEBUG: tempファイル削除失敗: {e}")
                            # キャッシュ保存
                            cache.set(f'results_{job_id}', results, timeout=3600)
                            cache.set(f'error_{job_id}', error, timeout=3600)

                thread = threading.Thread(target=process_files)
                thread.start()

                return JsonResponse({
                    'success': True,
                    'job_id': job_id,
                    'message': 'アップロードを受け付けました。現在OCR処理中です。'
                })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    # 通常GET時
    context = {
        "results": results,
        "error": error,
    }
    return render(request, "index.html", context)

@login_required
def export_receipts_csv(request):
    if request.user.is_superuser:
        receipts = Receipt.objects.all()
    else:
        receipts = Receipt.objects.filter(user=request.user)

    # BOM付きUTF-8
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="receipts_{timezone.now().strftime("%Y%m%d")}.csv"'
    response.write('\ufeff')  # ここがポイント！

    writer = csv.writer(response)
    writer.writerow(['ID', 'ユーザー', 'ファイル名', 'アップロード日時', 'OCRテキスト'])

    for receipt in receipts:
        writer.writerow([
            receipt.id,
            receipt.user.username,
            receipt.file.name,
            receipt.uploaded_at.strftime('%Y-%m-%d %H:%M'),
            receipt.text or ''
        ])
    return response

@login_required
@never_cache
def dashboard(request):
    # 一般ユーザーは自分のレシートだけ、管理者は全件
    if request.user.is_superuser:
        receipts = Receipt.objects.all()
    else:
        receipts = Receipt.objects.filter(user=request.user)
    total_count = receipts.count()

    # 新しい `calculated_total` プロパティを使って合計金額を計算
    total_amount = sum(r.calculated_total for r in receipts)

    # 月別集計
    monthly = (
        receipts
        .annotate(month=TruncMonth('uploaded_at'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    for m in monthly:
        # 月ごとのレシートを取得し、`calculated_total`で合計
        monthly_receipts = receipts.filter(uploaded_at__month=m['month'].month)
        m['total'] = sum(r.calculated_total for r in monthly_receipts)

    # カテゴリ別集計
    category_summary = (
        receipts
        .values('category')
        .annotate(count=Count('id'))
        .order_by('category')
    )
    for c in category_summary:
        # カテゴリごとのレシートを取得し、`calculated_total`で合計
        category_receipts = receipts.filter(category=c['category'])
        c['total'] = sum(r.calculated_total for r in category_receipts)

    # グラフ用データ
    category_labels = [c['category'] or "未分類" for c in category_summary]
    category_data = [c['total'] for c in category_summary]

    # 明細一覧
    receipt_summaries = []
    for receipt in receipts.order_by('-uploaded_at'):
        receipt_summaries.append({
            "id": receipt.id,
            "uploaded_at": receipt.uploaded_at,
            "user": receipt.user.username,
            "shop_name": getattr(receipt, "shop_name", ""),
            "store_name": getattr(receipt, "store_name", ""),
            "memo": getattr(receipt, "memo", ""),
            "total": receipt.calculated_total,
            "category": getattr(receipt, "category", "その他"),
        })

    return render(request, 'dashboard.html', {
        'total_count': total_count,
        'total_amount': total_amount,
        'monthly': monthly,
        'category_summary': category_summary,
        'category_labels': category_labels,
        'category_data': category_data,
        'category_labels_json': json.dumps(category_labels, ensure_ascii=False),
        'category_data_json': json.dumps(category_data, ensure_ascii=False),
        'receipt_summaries': receipt_summaries,
    })

def upload_view(request):
    if request.method == 'POST':
        job_id = request.POST.get('job_id')
        files = request.FILES.getlist('receipt')
        total = len(files)
        set_progress(job_id, total, 0)
        done = 0
        for f in files:
            # ここでOCRや保存処理
            done += 1
            set_progress(job_id, total, done)
            return HttpResponse('ファイル受信OK')
    return render(request, 'index.html')

def load_store_keywords():
    """店舗キーワードのJSONファイルを読み込む"""
    json_path = Path(__file__).parent / 'static' / 'store_keywords.json'
    try:
        with open(json_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"店舗キーワードJSONの読み込みに失敗: {str(e)}")
        return {}

# グローバル変数として読み込む
STORE_KEYWORDS = load_store_keywords()

print("[DEBUG] store_keywords_dict:", STORE_KEYWORDS)

def extract_product_items(text):
    items = extract_items(text)
    if items:
        return items
    total = extract_total(text)
    store_name = extract_store_name(text, STORE_KEYWORDS)
    # STORE_KEYWORDSを渡す
    category = guess_category_by_store(store_name, STORE_KEYWORDS)
    if total:
        return [{
            "name": "合計",
            "price": total,
            "category": category
        }]
    return []

@login_required
def receipt_dashboard(request):
    # 一般ユーザーは自分のレシートだけ、管理者は全件
    if request.user.is_superuser:
        receipts = Receipt.objects.all().order_by('-uploaded_at')
    else:
        receipts = Receipt.objects.filter(user=request.user).order_by('-uploaded_at')
    
    paginator = Paginator(receipts, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 合計件数・合計金額
    total_count = receipts.count()
    total_amount = 0
    for receipt in receipts:
        items = receipt.product_items.all()
        if items.exists():
            total_amount += sum([item.price for item in items])
        else:
            total_amount += getattr(receipt, 'total_amount', 0) or 0

    # カテゴリ別集計
    category_summary = (
        receipts.values('category')
        .annotate(count=Count('id'))
        .order_by('category')
    )
    for c in category_summary:
        c['total'] = 0
        for receipt in receipts.filter(category=c['category']):
            items = receipt.product_items.all()
            if items.exists():
                c['total'] += sum([item.price for item in items])
            else:
                c['total'] += getattr(receipt, 'total_amount', 0) or 0

    # グラフ用データ
    category_labels = [c['category'] for c in category_summary]
    category_data = [c['total'] for c in category_summary]
    
    return render(request, 'receipts/dashboard_receipts.html', {
        'receipts': page_obj,
        'page_obj': page_obj,
        'total_count': total_count,
        'total_amount': total_amount,
        'category_summary': category_summary,
        'category_labels': category_labels,
        'category_data': category_data,
    })

@login_required
def receipt_edit(request, receipt_id):
    # 一般ユーザーは自分のレシートだけ、管理者は全件
    if request.user.is_superuser:
        receipt = get_object_or_404(Receipt, id=receipt_id)
    else:
        receipt = get_object_or_404(Receipt, id=receipt_id, user=request.user)
    
    if request.method == 'POST':
        # 変更前の値を保存
        original_data = {
            'text': receipt.text,
            'shop_name': receipt.shop_name,
            'store_name': receipt.store_name,
            'memo': receipt.memo,
            'category': receipt.category,
            'total_amount': receipt.total_amount,
            'issue_date': receipt.issue_date,
        }
        
        # 商品明細の変更前の値も保存
        original_items = {}
        for item in receipt.product_items.all():
            original_items[item.id] = {
                'name': item.name,
                'price': item.price,
            }
        
        # OCRテキストから商品明細を再抽出
        if request.POST.get('reextract_items'):
            # --- ロック状態のチェック ---
            locked_keys = [k for k in request.POST.keys() if k.startswith('product_locked_')]
            locked_values = [request.POST.get(k) for k in locked_keys]
            all_locked = locked_values and all(v == '1' for v in locked_values)
            # OCRテキストから商品明細を抽出
            items = extract_product_items(receipt.text or "")
            # 商品明細をテキスト化
            items_text = "\n".join([f"{item['name']} {item['price']}円" for item in items]) if items else ''
            # --- 全てロック中なら商品明細は上書きせず、OCRテキスト欄のみ上書き ---
            if all_locked:
                receipt.text = items_text
                receipt.save()
                messages.info(request, '全ての金額が保護されているため、商品明細は変更せずOCRテキストのみ更新しました。')
                return redirect('receipt_edit', receipt_id=receipt.id)
            # --- 1つでもロック解除されていれば通常通り再抽出・上書き ---
            # 既存の明細を削除
            receipt.product_items.all().delete()
            for item in items:
                ProductItem.objects.create(
                    receipt=receipt,
                    name=item["name"],
                    price=item["price"],
                    category=item.get("category", "その他")
                )
            receipt.text = items_text
            receipt.save()
            messages.success(request, 'OCRテキストから商品明細を再抽出しました。')
            return redirect('receipt_edit', receipt_id=receipt.id)
        
        # フォームデータの処理
        # 一般ユーザーはユーザー変更を許可しない
        if request.user.is_superuser:
            user_id = request.POST.get('user')
            if user_id:
                try:
                    User = get_user_model()
                    user = User.objects.get(id=user_id)
                    receipt.user = user
                except User.DoesNotExist:
                    messages.error(request, '指定されたユーザーが見つかりません。')
                    return redirect('dashboard_receipts')
        
        receipt.text = request.POST.get('text', '')
        shop_name_post = request.POST.get('shop_name', None)
        if shop_name_post not in [None, '']:
            receipt.shop_name = shop_name_post
        store_name_post = request.POST.get('store_name', None)
        if store_name_post not in [None, '']:
            receipt.store_name = store_name_post
        receipt.memo = request.POST.get('memo', '')
        receipt.category = request.POST.get('category', 'その他')
        
        # ファイルのアップロード処理
        file_changed = False
        if 'file' in request.FILES:
            receipt.file = request.FILES['file']
            file_changed = True
        
        # 商品明細(ProductItem)の更新
        items_changed = False
        for item in receipt.product_items.all():
            name_key = f'product_name_{item.id}'
            price_key = f'product_price_{item.id}'
            new_name = request.POST.get(name_key, item.name)
            new_price_raw = request.POST.get(price_key, None)
            
            try:
                if new_price_raw is None or new_price_raw == '':
                    new_price = item.price
                else:
                    new_price = int(new_price_raw)
                    if new_price < 0:
                        new_price = item.price
            except Exception as e:
                print(f"[DEBUG] 金額変換エラー: {e}, item.id={item.id}, 入力値={new_price_raw}")
                new_price = item.price
            
            # 変更があったかチェック
            if (new_name != original_items[item.id]['name'] or 
                new_price != original_items[item.id]['price']):
                items_changed = True
            
            item.name = new_name
            item.price = new_price
            item.save()
        
        # 合計金額の保存処理をロック状態で分岐
        total_amount_post = request.POST.get('total_amount', None)
        locked = request.POST.get('total_amount_locked', '1')
        total_amount_changed = False
        if locked == '0' and total_amount_post not in [None, '']:
            try:
                new_total = int(total_amount_post)
                if new_total != original_data['total_amount']:
                    total_amount_changed = True
                receipt.total_amount = new_total
            except Exception:
                pass
        
        issue_date_post = request.POST.get('issue_date', None)
        issue_date_changed = False
        if issue_date_post not in [None, '']:
            try:
                if issue_date_post != str(original_data['issue_date'] or ''):
                    issue_date_changed = True
                receipt.issue_date = issue_date_post
            except Exception:
                pass
        
        # 変更があったかチェック
        data_changed = (
            receipt.text != original_data['text'] or
            receipt.shop_name != original_data['shop_name'] or
            receipt.store_name != original_data['store_name'] or
            receipt.memo != original_data['memo'] or
            receipt.category != original_data['category'] or
            file_changed or
            items_changed or
            total_amount_changed or
            issue_date_changed
        )
        
        receipt.save()
        
        if data_changed:
            messages.success(request, 'レシートが正常に更新されました。')
        else:
            messages.info(request, '変更はありませんでした。')
        
        return redirect('dashboard_receipts')
    
    # ユーザー一覧を取得（管理者のみ）
    users = []
    if request.user.is_superuser:
        User = get_user_model()
        users = User.objects.all()
    
    # カテゴリ選択肢
    categories = [
        '食費', '交通費', '日用品', '医療費', '娯楽費', 
        '衣類費', '光熱費', '通信費', 'その他'
    ]
    
    # 合計金額（total_amountがあればそれを優先）
    if getattr(receipt, 'total_amount', None) not in [None, '', 0]:
        total_amount = receipt.total_amount
    else:
        items = receipt.product_items.all()
        if items.exists():
            total_amount = sum([item.price for item in items])
        else:
            total_str = extract_total(receipt.text or "")
            try:
                total_amount = int(total_str)
            except Exception:
                total_amount = ''

    context = {
        'receipt': receipt,
        'users': users,
        'categories': categories,
        'total_amount': total_amount,
    }
    
    return render(request, 'receipts/receipt_edit.html', context)

@csrf_exempt
def receipt_delete(request, pk):
    if request.method == 'POST':
        # ログインユーザーのレシートであるか、またはスーパーユーザーであるかを確認
        receipt = get_object_or_404(Receipt, pk=pk)
        if receipt.user == request.user or request.user.is_superuser:
            receipt.delete()
            messages.success(request, '領収書を削除しました。')
            return redirect('dashboard_receipts')
        else:
            messages.error(request, '権限がありません。')
            return redirect('dashboard_receipts')
            
    messages.error(request, '無効なリクエストです。')
    return redirect('dashboard_receipts')

@login_required
def receipt_bulk_delete(request):
    if request.method == "POST":
        ids = request.POST.getlist('ids')
        if not ids:
            messages.warning(request, '削除する項目が選択されていません。')
            return redirect('dashboard_receipts')

        if request.user.is_superuser:
            deleted_count, _ = Receipt.objects.filter(id__in=ids).delete()
        else:
            deleted_count, _ = Receipt.objects.filter(id__in=ids, user=request.user).delete()
        
        if deleted_count > 0:
            messages.success(request, f'{deleted_count}件の領収書を一括削除しました。')
        else:
            messages.warning(request, '削除できる領収書がありませんでした。')

        return redirect('dashboard_receipts')

    messages.error(request, '無効なリクエストです。')
    return redirect('dashboard_receipts')

@login_required
def receipt_create(request):
    if request.method == 'POST':
        form = ReceiptForm(request.POST, request.FILES)
        if form.is_valid():
            receipt = form.save(commit=False)
            receipt.user = request.user  # ログインユーザーを自動セット
            receipt.save()
            return redirect('receipt_list')  # 登録後のリダイレクト先
    else:
        form = ReceiptForm()
    return render(request, 'receipts/receipt_create.html', {'form': form})

def receipt_list(request):
    receipts = Receipt.objects.all()
    return render(request, 'receipts/receipt_list.html', {'receipts': receipts})

@login_required
def dashboard_summary_api(request):
    """ダッシュボードのサマリー情報を返すAPI"""
    user = request.user
    if user.is_staff:
        receipts = Receipt.objects.all()
    else:
        receipts = Receipt.objects.filter(user=user)

    summary = receipts.aggregate(
        total_amount=Sum('total_amount'),
        receipt_count=Count('id')
    )
    return JsonResponse({
        'total_amount': summary['total_amount'] or 0,
        'receipt_count': summary['receipt_count'] or 0,
    })

def privacy_policy_view(request):
    """プライバシーポリシーページを表示するビュー"""
    return render(request, 'privacy.html')

def terms_of_service_view(request):
    """利用規約ページを表示するビュー"""
    return render(request, 'terms.html')

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'アカウントが正常に作成されました。ログインしてください。')
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

@login_required
def upload_progress(request):
    job_id = request.GET.get('job_id')
    progress = get_progress(job_id)
    return JsonResponse(progress)

@login_required
def receipt_list_partial(request):
    # 最新の領収書リストを取得
    receipts = Receipt.objects.filter(user=request.user).order_by('-uploaded_at')[:20]
    html = render_to_string('partials/receipt_list.html', {'receipts': receipts})
    return JsonResponse({'html': html})

# 進捗セット・取得用関数
def set_progress(job_id, total, done):
    cache.set(f'progress_{job_id}', {'total': total, 'done': done}, timeout=3600)

def get_progress(job_id):
    return cache.get(f'progress_{job_id}', {'total': 0, 'done': 0})

@login_required
def get_processing_results(request):
    """処理結果を取得するAPI"""
    job_id = request.GET.get('job_id')
    
    if job_id:
        results = cache.get(f'results_{job_id}', [])
        print(f"[DEBUG] get-results job_id={job_id}, results={results}")
        error = cache.get(f'error_{job_id}', None)
        
        return JsonResponse({
            'success': True,
            'results': results,
            'error': error
        })
    
    return JsonResponse({
        'success': False,
        'error': 'Job ID not provided'
    })

def ensure_bytes(content):
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        # Data URI 形式
        if content.startswith("data:image"):
            _, base64_data = content.split(",", 1)
            return base64.b64decode(base64_data)
        # base64文字列
        try:
            return base64.b64decode(content)
        except Exception:
            raise ValueError("Content is a string but not valid base64")
    raise TypeError("Unsupported image content type")
