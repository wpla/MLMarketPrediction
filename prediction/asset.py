import pandas as pd
from datetime import datetime

from log import Log


class Asset:
    def __init__(self):
        self.data = None
        self.name = None
        self.symbol = None
        self.exchange = None
        self.header_lines = None

    def read_header(self, filename):
        self.header_lines = 0
        with open(filename) as file:
            head = [next(file) for n in range(3)]
        for line, nr in zip(head, range(1, 4)):
            parts = line.strip().split(":")
            if len(parts) != 2:
                break
            self.header_lines = nr
            key, value = [part.strip() for part in parts]
            if key == "Symbol":
                self.symbol = value
            elif key == "Name":
                self.name = value
            elif key == "Exchange":
                self.exchange = value

    def read_csv(self, filename):
        self.read_header(filename)
        self.data = pd.read_csv(filename, skiprows=self.header_lines, sep=";", converters={0: lambda x: datetime.strptime(x, "%Y-%m-%d")})
        self.data = self.data.set_index('Date')

    def write_csv(self, filename):
        outfile = open(filename, "w")
        if self.symbol is not None:
            outfile.write("Symbol: %s\n" % self.symbol)
        if self.name is not None:
            outfile.write("Name: %s\n" % self.name)
        if self.exchange is not None:
            outfile.write("Exchange: %s\n" % self.exchange)
        self.data.to_csv(outfile, sep=";", line_terminator='\n')

    def append(self, col, series: pd.Series):
        self.data[col] = series



