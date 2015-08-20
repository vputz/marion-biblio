"""
Progress reporting as generator
"""
from gevent import sleep
from gevent.queue import Queue

# value to test when the report is complete
PROGRESS_COMPLETE = "progress_complete"

def funcProgressGenerator(gen, lengthHint, reportFunc):
    """
    Generates progress reports, using the function reportFunc

    gen: The generator used for iteration
    lengthHint: the "Best guess" length for gen; for most sequences,
      pass length.
    reportFunc: f( step, length ), function called on each step
    """
    step = 0
    for item in gen:
        step = step+1
        reportFunc(step, lengthHint)
        yield item
    reportFunc(PROGRESS_COMPLETE, lengthHint)


class NullProgressReporter:
    """
    ProgressReporters have one fixed method: reportProgress.
    This one does nothing.
    It is assumed that length hints are taken into account
    in the initialization of the reporter.
    """

    def reportProgress(self, step):
        pass


NPG = NullProgressReporter()


class StringBuilderReporter:

    def __init__(self, lengthHint):
        self.lengthHint = lengthHint
        self.status = ""

    def reportProgress(self, step):
        self.status = str(step)+"/"+str(self.lengthHint)


class QueueReporter:

    def __init__(self, length_hint):
        self.queue = Queue()
        self.length_hint = length_hint

    def reportProgress(self, step):
        self.queue.put((step, self.length_hint))
        sleep(0)


def reporterProgressGenerator(gen, reporter):

    step = 0
    for item in gen:
        step = step+1
        reporter.reportProgress(step)
        yield item
    reporter.reportProgress(PROGRESS_COMPLETE)
