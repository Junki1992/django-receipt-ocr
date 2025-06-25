# from celery import shared_task
from django.contrib.auth import get_user_model
from django.core.files import File
from django.conf import settings
from .models import Receipt
from .util import (
    STORE_KEYWORDS, extract_store_name, guess_category_by_store, 
    extract_total, is_receipt_image, ocr_google_vision
)
import os
import uuid

@shared_task
def process_uploaded_files(job_id, saved_files, user_id):
    """アップロードされたファイルを非同期で処理"""
    User = get_user_model()
    user = User.objects.get(id=user_id)
    results = []
    done = 0
    total = len(saved_files)
    
    for file_info in saved_files:
        try:
            temp_path = file_info['path']
            original_name = file_info['original_name']
            is_from_zip = file_info['is_from_zip']
            
            temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
            
            # ファイル処理
            if not is_receipt_image(temp_full_path):
                results.append({
                    "filename": original_name,
                    "status": "NG",
                    "message": "この画像はレシート・領収書として認識できませんでした。"
                })
                done += 1
                set_progress(job_id, total, done)
                continue
            
            # OCR処理
            text = ocr_google_vision(temp_full_path)
            
            # 店舗名、カテゴリ、合計金額を抽出
            store_name = extract_store_name(text, STORE_KEYWORDS)
            category = guess_category_by_store(store_name, STORE_KEYWORDS)
            total_amount_str = extract_total(text)
            total_int = int(total_amount_str) if total_amount_str.isdigit() else 0
            
            # 拡張子を取得
            ext = os.path.splitext(original_name)[1].lower()
            
            # DB保存
            unique_name = f"{uuid.uuid4()}{ext}"
            receipt = Receipt.objects.create(
                file=File(open(temp_full_path, 'rb'), name=unique_name),
                user=user,
                store_name=store_name,
                category=category,
                text=text,
                total_amount=total_int,
            )
            
            # 商品アイテム抽出タスクを実行
            run_ocr_task.delay(receipt.id)
            
            results.append({
                "filename": original_name,
                "status": "OK",
                "message": "アップロード成功",
            })
            
        except Exception as e:
            results.append({
                "filename": original_name,
                "status": "NG",
                "message": f"アップロード失敗: {str(e)}",
            })
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_full_path):
                os.remove(temp_full_path)
            
            done += 1
            set_progress(job_id, total, done)
    
    return results

@shared_task
def run_ocr_task(receipt_id):
    from .views import ocr_google_vision, extract_store_name, extract_total, STORE_KEYWORDS, guess_category_by_store
    receipt = Receipt.objects.get(id=receipt_id)
    text = ocr_google_vision(receipt.file.path)
    store_name = extract_store_name(text, STORE_KEYWORDS)
    category = guess_category_by_store(store_name, STORE_KEYWORDS)
    total_amount_str = extract_total(text)
    total_int = int(total_amount_str) if total_amount_str.isdigit() else 0

    receipt.text = text
    receipt.store_name = store_name
    receipt.category = category
    receipt.total_amount = total_int
    receipt.save()

def set_progress(job_id, total, done):
    """進捗を保存"""
    if job_id:
        from django.core.cache import cache
        cache.set(f"progress_{job_id}", {"total": total, "done": done}, timeout=3600)
