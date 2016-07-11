__author__ = 'admin'
import unittest
import matplotlib.pyplot as plt


class PlotTest(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def plot_xy_test(self):
        # simplest xy plot
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
        plt.show()

    def multiplot_xy_test(self):
        # multiline plots
        x = [i for i in range(10)]
        y1 = [i for i in range(10)]
        y2 = [2 * i for i in range(10)]
        plt.plot(x, y1, 'r--', x, y2, 'bs')
        plt.show()

    def multiplot_xy_test2(self):
        # multiline plots
        x = [i for i in range(10)]
        y1 = [i for i in range(10)]
        plt.plot(x, y1, 'r--')

        y2 = [2 * i for i in range(10)]
        plt.plot(x, y2, 'bs')

        plt.show()


    def legend_test(self):
        # x = [1, 2, 3, 4]
        # y = [1, 4, 9, 16]
        # plt.plot(x, y, 'ro', label='square')
        # plt.legend(loc='upper left', frameon=False)
        # plt.show()

        # multiline plots
        x = [i for i in range(10)]
        y1 = [i for i in range(10)]
        y2 = [2 * i for i in range(10)]
        plt.plot(x, y1, 'r--', label='y1 line')
        plt.plot(x, y2, 'ro', label='y2 dot')
        plt.legend(loc='upper left', frameon=False)
        plt.show()