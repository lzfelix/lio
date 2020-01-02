import re
import csv
from typing import List


def open_table(writer_fn: callable) -> None:
    writer_fn(r'\begin{table*}[]')
    writer_fn(r'\centering')
    writer_fn(r'\caption{Best fitness found by varying the number of decision variables for each benchmark function.}')
    writer_fn(r'\label{t.results}')
    writer_fn(r'\renewcommand{\arraystretch}{1.5}')
    writer_fn(r'\resizebox{\textwidth}{!}{')
    writer_fn(r'\begin{tabular}{lcrrlrr}')
    writer_fn(r'\hline')

    col_names = ['Functions', '\# Dimensions', 'QPSO', 'QPSO + LIO', 'p', 'QPSO time (s)', 'LIO time (s)']
    header = ' &'.join([f"\\multicolumn{{1}}{{c}}{{\\textbf{{{col}}}}}" for col in col_names])
    writer_fn(header)
    writer_fn(r'\\ \hline')


def close_table(writer_fn: callable) -> None:
    writer_fn(r'\hline')
    writer_fn(r'\end{tabular}}')
    writer_fn(r'\end{table*}')


def latex_line(writer_fn: callable, line: dict, first_line: bool = False) -> None:
    if first_line:
        writer_fn(r'\hline')
        first_line = f"\multirow{{4}}{{*}}{{{line['fn'].capitalize()}}} & \n"
    else:
        first_line = "& "

    default_format = "\t${:.4e} \\pm {:.4e}$\t & \n"
    bold_format = "\t$\\bm{{{:.4e} \\pm {:.4e}}}$\t & \n"
    time_format = "\t{:.2f} $\\pm$ {:.2f}\t"

    if float(line['wilcoxon']) > 0.05:
        qpso_format = lio_format = bold_format
    else:
        if float(line['qpso']) < float(line['lio']):
            qpso_format = bold_format
            lio_format = default_format
        else:
            qpso_format = default_format
            lio_format = bold_format

    writer_fn(
        first_line +
        f"{line['dims']:3}\t & \n" +
        qpso_format.format(float(line['qpso']), float(line['qpso_std'])) +
        lio_format.format(float(line['lio']), float(line['lio_std'])) +
        f"\t{line['p']:4.4} $\pm$ {line['p_std']:4.4}\t & \n" +
        time_format.format(float(line['qpso_time']), float(line['qpso_time_std'])) + '&\n' +
        time_format.format(float(line['lio_time']), float(line['lio_time_std'])) + '\\\\'
    )


def fix_notation(all_entries: List[str]) -> List[str]:
    """Replaces e notation by power-of-ten notation"""
    return [re.sub(r'e[\+\-](\d+)', lambda x: f' \\cdot 10^{{{int(x.group(1))}}}', line) for line in all_entries]


if __name__ == '__main__':
    sorted_fns = ['sphere', 'csendes', 'salomon', 'ackley1', 'alpine1', 'rastrigin', 'schwefel', 'brown']
    csv_fields = ['fn', 'dims', 'qpso', 'qpso_std', 'qpso_time', 'qpso_time_std',
                  'lio', 'lio_std', 'lio_time', 'lio_time_std',
                  'p', 'p_std', 'wilcoxon']

    entries = []
    with open('./results.csv') as dfile:
        reader = csv.DictReader(dfile, csv_fields)

        current_bucket = []
        fn_entries = dict()
        previous_fn_name = None

        next(reader)  # Skipping the header
        for line in reader:
            flag = False
            fn_name = line['fn']
            # previous_fn_name = previous_fn_name or fn_name

            if fn_name != previous_fn_name:
                fn_entries[previous_fn_name] = current_bucket[:]
                previous_fn_name = fn_name
                current_bucket = []
                flag = True
            latex_line(current_bucket.append, line, flag)

        # Appending the last fn entry
        fn_entries[previous_fn_name] = current_bucket[:]

    # Putting csv entries in the order presented in the paper:
    sorted_entries = []
    for fn_name in sorted_fns:
        sorted_entries.extend(fn_entries[fn_name])

    # Making everything a valid latex text
    sorted_entries = fix_notation(sorted_entries)
    sorted_entries = '\n'.join(sorted_entries)



    # There you go
    open_table(print)
    print(sorted_entries)
    close_table(print)
