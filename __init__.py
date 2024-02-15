# __init__.py dosyası paket yapılandırması
import logging

# Paket başlatıldığında bir log kaydı oluştur
logging.basicConfig(level=logging.INFO)

# Dışa aktarılacak modülleri tanımla
from .module1 import some_function
from .module2 import another_function