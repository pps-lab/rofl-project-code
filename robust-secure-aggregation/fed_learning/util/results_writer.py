import csv

RESULTS_DOC = 'server_eval.csv'

def append(round, loss, accuracy, time):
    with open(RESULTS_DOC,'a') as fd:
        writer = csv.writer(fd)
        writer.writerow([round, loss, accuracy, time])
