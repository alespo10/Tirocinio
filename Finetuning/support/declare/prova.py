from declare4py.declare4py import Declare4Py

# Percorso del log in formato XpyES (standard per i log di eventi)
log_path = "/Users/alessandro/PycharmProjects/Tirocinio/Preprocessing/Input/helpdesk.csv"

# Inizializza il motore DECLARE
d4py = Declare4Py()
d4py.parse_xes_log(log_path)

# Estrai i vincoli automaticamente
constraints = d4py.mine_constraints()
for constraint in constraints:
    print(constraint)
