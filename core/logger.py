class _Formatter:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def bold(cls, message: str) -> str:
        return cls.BOLD + message + cls.ENDC

    @classmethod
    def green(cls, message: str) -> str:
        return cls.OKGREEN + message + cls.ENDC

    @classmethod
    def blue(cls, message: str) -> str:
        return cls.OKBLUE + message + cls.ENDC

    @classmethod
    def yellow(cls, message: str) -> str:
        return cls.WARNING + message + cls.ENDC

    @classmethod
    def red(cls, message: str) -> str:
        return cls.FAIL + message + cls.ENDC


class Logger:

    muteInfo = False
    muteDebug = False
    muteWarning = False
    muteError = False
    muteAll = False

    def __init__(self, name: str):
        self.name = name

    def info(self, message):
        name = _Formatter.bold('[' + self.name + ']')
        name = _Formatter.green(name)
        level = _Formatter.green('[INFO]')
        message = _Formatter.green(message)

        if not self.muteAll and not self.muteInfo:
            print(name + level + ' ' + message)

    def debug(self, message):
        name = _Formatter.bold('[' + self.name + ']')
        name = _Formatter.blue(name)
        level = _Formatter.blue('[DEBUG]')
        message = _Formatter.blue(message)

        if not self.muteAll and not self.muteDebug:
            print(name + level + ' ' + message)

    def warning(self, message):
        name = _Formatter.bold('[' + self.name + ']')
        name = _Formatter.yellow(name)
        level = _Formatter.yellow('[WARNING]')
        message = _Formatter.yellow(message)

        if not self.muteAll and not self.muteWarning:
            print(name + level + ' ' + message)

    def error(self, message):
        name = _Formatter.bold('[' + self.name + ']')
        name = _Formatter.red(name)
        level = _Formatter.red('[ERROR]')
        message = _Formatter.red(message)

        if not self.muteAll and not self.muteError:
            print(name + level + ' ' + message)
