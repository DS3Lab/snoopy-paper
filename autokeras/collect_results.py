from os import listdir
from os.path import isfile, join
import json

mypath = "outputs"

datasets = [d for d in listdir(mypath)]

results = {}
for d in datasets:
    print("Dataset", d)
    files = [f for f in listdir(join(mypath, d)) if isfile(join(mypath, d, f)) and f.endswith(".log")]
    print("Number of log files:", len(files))

    for f in files:
        noise = float(f.split("_noise_")[1].split("_")[0])
        lr = float(f.split("_lr_")[1].split("_")[0])
        reg = float(f.split("_reg_")[1].split("_")[0])
        acc = None
        time = None
        with open(join(mypath, d, f)) as fs:
            lines = fs.readlines()
            for l in lines:
                str1 = "accuracy: "
                if l.startswith(str1):
                    acc = float(l.split(str1)[1])
                    continue
                str2 = "time: "
                if l.startswith(str2):
                    time = float(l.split(str2)[1])
                    continue
                if acc is not None and time is not None:
                    break
        assert acc is not None and time is not None

        if d not in results.keys():
            results[d] = {}

        if noise not in results[d].keys():
            results[d][noise] = {}

        hpkey = "({}, {})".format(reg, lr)

        if hpkey not in results[d][noise].keys():
            results[d][noise][hpkey] = {}
            results[d][noise][hpkey]["errors"] = []
            results[d][noise][hpkey]["times"] = []

        results[d][noise][hpkey]["errors"].append(1.0 - acc)
        results[d][noise][hpkey]["times"].append(time)

for d, v1 in sorted(results.items()):
    print("Dataset:", d)
    res = []
    times = []
    for noise, v2 in sorted(v1.items()):
        print(" Noise:", noise)
        min_error = min([sum(v['errors'])/float(len(v['errors'])) for k, v in v2.items()])
        times = [sum(v['times'])/float(len(v['times'])) for k, v in v2.items()]
        print(" Error", min_error)
        print(" Avg-time:", sum(times)/float(len(times)))
