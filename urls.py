from django.urls import include, path, re_path
from django.conf.urls import url
#from .api import views as api_views
#from .views import description, input, output, algorithms, references, qaqc
#from .views import misc, landing
#from .views import batch
#from .views import sam_watershed
#from .views import proxy
from .views import input

print('qed.nta_app.urls')

urlpatterns = [
    # django 2.X
    path('', input.input_page),
    path('input/', input.input_page),
    #path('output/',  ),

]

# 404 Error view (file not found)
#handler404 = misc.file_not_found
# 500 Error view (server error)
#handler500 = misc.file_not_found
# 403 Error view (forbidden)
#handler403 = misc.file_not_found