import matplotlib.pyplot as plt
import sys

file_path = "results/xia3 5000 iter/output.bot"

if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"output"

with open(file_path, "rb") as file:
    ipopt = file.readlines()[24:]

stats = {}
keys = list(filter(None, list(str(ipopt[0]).split(" "))))[0:4]
keys[0] = keys[0].replace("b'", "")
for key in keys:
    stats[key] = []

i = 15
while True:
    if len(list(filter(None, list(str(ipopt[-i]).split(" "))))) >= 10:
        break
    i += 1
ipopt = ipopt[1 : (len(ipopt) - i)]

for k in ipopt:
    line = list(filter(None, list(str(k).replace("b'","").split(" "))))
    if "iter" not in line[0]:
        if "r" in line[0]:
            line[0] = line[0].replace("r", "")
        for k, key in enumerate(keys):
            stats[key].append(float(line[k]))

for k, key in enumerate(keys[1:]):
    plt.plot(stats[keys[0]], stats[key], "-o", markersize=1, label=key, color=["b", "g", "r"][k])
plt.yscale("log")
plt.legend()
plt.show()
