"""A simple command-line tool to convert WOS data from tab-delimited to H5"""
from marion_biblio.wos_reader import make_pytable, open_wos_tab
import argparse
import gevent
from marion_biblio.progressivegenerators import QueueReporter

# strictly speaking argparse isn't important here but including it
# for when we get all fancy like eliminating abstracts etc
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert csv files to h5 for WOS")
    parser.add_argument('files', type=str, nargs='+',
                        help="CSV files to be added to the h5")
    parser.add_argument('-o', '--out', type=str, help='Output file',
                        required=True)
    parser.add_argument('-t', '--title', type=str,
                        help='Title of the output h5 (optional)',
                        default='Unnamed')
    args = parser.parse_args()
    with open_wos_tab(args.files) as wos:

        reporter = QueueReporter(length_hint=0)

        def do_action():
            make_pytable(wos, args.out, args.title,
                         progressReporter=reporter)
        g1 = gevent.spawn(do_action)

        def report():
            while not g1.dead:
                while not reporter.queue.empty():
                    print(reporter.queue.get())
                gevent.sleep(0)

        gevent.joinall([
            g1,
            gevent.spawn(report)
        ])
