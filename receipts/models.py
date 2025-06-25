from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth import get_user_model as get_user_model_auth

# Create your models here.

class Receipt(models.Model):
    # 領収書ファイル
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, null=True)
    file = models.FileField(upload_to='receipts/')
    # アップロード日時
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # OCRで抽出した全文
    text = models.TextField(blank=True, null=True)
    # 編集用の店舗名（ユーザーが自由に入力・修正できる）
    shop_name = models.CharField(max_length=255, blank=True, null=True)
    # ロゴ判定・OCRで推定した店舗名
    store_name = models.CharField(max_length=255, blank=True, null=True)
    # ロゴ判定・OCRで推定したカテゴリ（例：飲食、買い物、移動、交際費）
    category = models.CharField(max_length=100, blank=True, null=True, default="その他")
    logo_confidence = models.FloatField(default=0.0)  # ロゴ検出の信頼度
    total_amount = models.IntegerField(blank=True, null=True)
    purchase_date = models.DateTimeField(blank=True, null=True)
    memo = models.TextField(blank=True, null=True)
    issue_date = models.DateField(null=True, blank=True, verbose_name="発行日")

    def __str__(self):
        return f"Receipt {self.id} by {self.user.username}"

    @property
    def calculated_total(self):
        """
        ProductItemがあればその合計を、なければtotal_amountを返すプロパティ
        """
        items = self.product_items.all()
        if items.exists():
            return sum(item.price for item in items)
        return self.total_amount or 0

class ProductItem(models.Model):
    # 紐づく領収書
    receipt = models.ForeignKey(Receipt, related_name='product_items', on_delete=models.CASCADE)
    # 商品名
    name = models.CharField(max_length=255)
    # 金額
    price = models.IntegerField()
    # カテゴリ
    category = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.name} - {self.price}円"
