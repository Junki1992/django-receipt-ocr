from django.urls import path
from receipts import views
from receipts.views import receipt_dashboard  # 必要な関数だけ
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('export/csv/', views.export_receipts_csv, name='export_receipts_csv'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/receipts/', views.receipt_dashboard, name='dashboard_receipts'),
    path('privacy/', views.privacy_policy_view, name='privacy'),
    path('terms/', views.terms_of_service_view, name='terms'),
    path('signup/', views.signup, name='signup'),
    path('receipt/<int:receipt_id>/edit/', views.receipt_edit, name='receipt_edit'),
    path('receipt/<int:pk>/delete/', views.receipt_delete, name='receipt_delete'),
    path('receipt/bulk_delete/', views.receipt_bulk_delete, name='receipt_bulk_delete'),
    path('create/', views.receipt_create, name='receipt_create'),
    path('list/', views.receipt_list, name='receipt_list'),
    path('api/dashboard_summary/', views.dashboard_summary_api, name='dashboard_summary_api'),
    path('get-results/', views.get_processing_results, name='get_processing_results'),
    path('password_change/', auth_views.PasswordChangeView.as_view(
        template_name='registration/password_change_custom.html'
    ), name='password_change'),
    path('password_change/done/', auth_views.PasswordChangeDoneView.as_view(
        template_name='registration/password_change_done_custom.html'
    ), name='password_change_done'),
]
