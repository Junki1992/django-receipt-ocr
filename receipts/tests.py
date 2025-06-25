from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from .models import Receipt
from django.core.files.uploadedfile import SimpleUploadedFile
import os
from .views import extract_product_items

# Create your tests here.

class ReceiptModelTest(TestCase):
    def test_create_receipt(self):
        user = get_user_model().objects.create_user(username='testuser', password='testpass')
        receipt = Receipt.objects.create(user=user, file='test.jpg')
        self.assertEqual(receipt.user.username, 'testuser')

class FileUploadTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='testuser', password='testpass')
        self.client = Client()
        self.client.login(username='testuser', password='testpass')

    def test_file_upload(self):
        # テスト用ファイルのパス
        test_file_path = os.path.join(os.path.dirname(__file__), 'test_receipt.jpg')
        with open(test_file_path, 'rb') as f:
            file_data = SimpleUploadedFile('test_receipt.jpg', f.read(), content_type='image/jpeg')
            response = self.client.post('/', {'file': file_data}, follow=True)
        self.assertEqual(response.status_code, 200)
        # 必要に応じてDBやレスポンス内容もチェック

class ProductExtractionTest(TestCase):
    def test_product_extraction(self):
        text = "チーズスモ 199 小計199 合計215"
        items = extract_product_items(text)
        # 商品名や金額が正しく抽出されているか検証
        self.assertTrue(any(item['name'] == 'チーズスモ' for item in items))
        self.assertTrue(any(item['price'] == 199 for item in items))

class PermissionTest(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user1 = User.objects.create_user(username='user1', password='pass1')
        self.user2 = User.objects.create_user(username='user2', password='pass2')
        self.client = Client()
        # user1でログインして領収書を登録
        self.client.login(username='user1', password='pass1')
        self.receipt = Receipt.objects.create(user=self.user1, file='test1.jpg')
        self.client.logout()

    def test_user_cannot_access_others_receipt(self):
        # user2でログイン
        self.client.login(username='user2', password='pass2')
        # user1の領収書詳細ページにアクセス（URLは実装に合わせて調整）
        response = self.client.get(f'/receipts/{self.receipt.id}/')
        # 403 Forbiddenや404 Not Foundなど、アクセスできないことを確認
        self.assertNotEqual(response.status_code, 200)

class DashboardTest(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client = Client()
        self.client.login(username='testuser', password='testpass')
        # 必要に応じて領収書・商品情報を作成
        # 例: Receipt.objects.create(user=self.user, file='test.jpg', ...)

    def test_dashboard_view(self):
        response = self.client.get('/dashboard/', follow=True)  # 実際のURLに合わせて調整
        self.assertEqual(response.status_code, 200)
        self.assertIn('合計金額', response.content.decode())  # 集計値やキーワードで確認
