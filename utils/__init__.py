from datetime import datetime
from . import ops

class Logger:
    def __init__(self, path, **kwargs):
        append = kwargs.get('append', False)
        self.file = open(path, 'a' if append else 'w') if path else None
        self.flush = kwargs.get('flush', True)

    def __call__(self, *args, **kwargs):
        if self.file:
            self.file.write(args[0] + '\n', *args[1:])
            if self.flush:
                self.file.flush()

        print(*args)

    def close(self):
        if self.file:
            self.file.close()

    def __delete__(self, instance):
        self.close()


class Profiler:
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger', print)
        self.label = kwargs.get('label', '')
        self.info = [None] * 3

    def __enter__(self):
        self.info[0] = datetime.now()
        return self.info

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.info[1] = datetime.now()
        self.info[2] = self.info[1] - self.info[0]

        if self.logger:
            label = ': {}'.format(self.label) if self.label else ''
            self.logger('[profiler{}] elapsed: {}'.format(label, self.info[2]))
