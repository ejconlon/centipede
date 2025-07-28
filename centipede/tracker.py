import curses


def loop(scr):
    scr.clear()

    for i in range(0, 11):
        v = i - 10
        scr.addstr(i, 0, "10 divided by {} is {}".format(v, 10 / v))
        scr.refresh()
        scr.getkey()


def main():
    curses.wrapper(loop)
