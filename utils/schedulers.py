from utils.conf import args

def tau_scheduler(acc):
    if acc > 0.95:
        return 0.1
    elif acc > 0.9:
        return 0.5
    elif acc > 0.8:
        return 1.0
    else:
        return args.tau