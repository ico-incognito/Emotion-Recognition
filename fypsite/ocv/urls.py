from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
#from ocv.views import example

urlpatterns = [
	#path('', example.as_view(), name='index'),
	path('', views.index, name='index'),
	#path('left/', views.left, name='left'),
	#path('right/', views.right, name='right'),
	path('opencv/', views.opencv, name='opencv'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)