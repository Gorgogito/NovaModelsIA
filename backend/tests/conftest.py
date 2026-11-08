import os
import tempfile

# Debe fijarse antes de importar nova.*: aísla la base de datos de las pruebas.
os.environ["NOVA_DATA"] = tempfile.mkdtemp(prefix="nova_test_")
