from subprocess import Popen
import numpy

for p_recall in numpy.linspace(0.1, 0.9, 9):
    python_interpreter = (
        "/home/juliengori/Documents/VC/imlm/paper/theory/.env/bin/python3"
    )
    script_name = "/home/juliengori/Documents/VC/imlm/paper/theory/schedules/effect_of_recall_p.py"
    cmd_string = f"{python_interpreter} {script_name} {p_recall}"
    process = Popen(cmd_string, shell=True)
