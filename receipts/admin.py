from django.contrib import admin
from django.utils.safestring import mark_safe
from .models import Receipt, ProductItem
from .util import extract_items, extract_total, extract_store_name, STORE_KEYWORDS

# サイトタイトル等のカスタマイズ（任意）
admin.site.site_header = "Receiptly 管理画面"
admin.site.site_title = "Receiptly Admin"
admin.site.index_title = "管理メニュー"

# すべてのModelAdminにCSSを適用
class GlobalAdmin(admin.ModelAdmin):
    class Media:
        css = {
            'all': ('css/admin_custom.css',)
        }

# ↓ ここにMyAdminSiteの定義を移動
class MyAdminSite(admin.AdminSite):
    site_header = "Receiptly 管理サイト"
    site_title = "Receiptly 管理"
    index_title = "ダッシュボード"

    class Media:
        css = {
            'all': ('css/admin_custom.css',)
        }

my_admin_site = MyAdminSite()

@admin.register(Receipt, site=my_admin_site)
class ReceiptAdmin(GlobalAdmin):
    list_display = ('id', 'user', 'uploaded_at', 'image_tag', 'short_text')
    readonly_fields = ('image_tag', 'text')
    search_fields = ('file', 'text')
    list_filter = ('uploaded_at',)

    def image_tag(self, obj):
        if obj.file and str(obj.file.url).lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            return mark_safe(
                f'<img src="{obj.file.url}" class="receipt-thumbnail" style="width:130px; height:220px; object-fit:cover; cursor:pointer;" />'
            )
        return "画像なし"
    image_tag.short_description = '画像プレビュー'

    def short_text(self, obj):
        text = obj.text or ""
        store = extract_store_name(text, STORE_KEYWORDS)
        total = extract_total(text)
        display = f"<div class='ocr-row'><b>店舗:</b> {store if store else '不明'}</div>"
        if total:
            display += f"<div class='ocr-row'><b>合計:</b> {total}円</div>"
        return mark_safe(f'<div style="max-width:300px;white-space:pre-wrap;overflow-wrap:break-word;word-break:break-all;text-align:left;">{display}</div>')
    short_text.short_description = 'OCRテキスト（整形）'

# @admin.register(ProductItem, site=my_admin_site)
# class ProductItemAdmin(GlobalAdmin):
#     list_display = ('id', 'receipt', 'name', 'price', 'category')
#     search_fields = ('name', 'category')
#     list_filter = ('category',)
