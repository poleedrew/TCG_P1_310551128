# 2584-Framework

Framework for 2584 & 2584-like Games (C++ 11)

## Basic Usage

To make the sample program:
```bash
make # see makefile for details
```

To run the sample program:
```bash
./2584 # by default the program runs 1000 games
```

To specify the total games to run:
```bash
./2584 --total=100000
```

To display the statistic every 1000 episodes:
```bash
./2584 --total=100000 --block=1000 --limit=1000
```

To specify the total games to run, and seed the environment:
```bash
./2584 --total=100000 --evil="seed=12345" # need to inherit from random_agent
```

To save the statistic result to a file:
```bash
./2584 --save=stat.txt
```

To load and review the statistic result from a file:
```bash
./2584 --load=stat.txt
```

## Author

[Computer Games and Intelligence (CGI) Lab](https://cgilab.nctu.edu.tw/), NYCU, Taiwan