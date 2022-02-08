import copy
import time
from random import randint
from pprint import pprint as pp

class Vector(list):
    def __init__(self, *args):
        super().__init__()
        for arg in args:
            self.append(arg)

    def __getitem__(self, index):
        return super()[index - 1]

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Vector(*[row * other for row in self])

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Vector(*[row / other for row in self])

    def __add__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Vector(*[row + other for row in self])

    def __sub__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Vector(*[row - other for row in self])

    def __str__(self):
        return "<" + ", ".join([str(e) for e in self]) + ">"


class Matrix:
    def __init__(self, matrix):
        self._list = copy.deepcopy(matrix)

    def __str__(self):
        rows, columns = len(self._list), len(self._list[1])
        max_length, elements = 0, []
        for row in self:
            for element in row:
                elements.append(element)
                if (l := len(str(element))) > max_length:
                    max_length = l

        max_length += 1
        string = ""
        for i, element in enumerate(elements):
            addition = f"{element}"
            string += " " * (max_length - len(addition)) + addition
            if (i + 1) % columns == 0:
                string += "\n"

        return string[:-1]

    def __iter__(self):
        return iter([self[i + 1] for i in range(len(self))])

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        # print(f"\"{index}\"")
        is_tuple = isinstance(index, tuple)
        length = 1 if not is_tuple else len(index)

        r, c = 0, 0
        if length == 1:  # Grabbing a Row
            if is_tuple:
                index = index[0]
            r = index

        elif length == 2:  # Grabbing a Row, a Column, or an Element
            r, c = index

        if c <= 0:  # Grabbing a Row
            return Vector(*self._list[r - 1])
        elif r <= 0:  # Grabbing a Column
            return Vector(*[self._list[i][c - 1] for i in range(len(self._list))])

        return self._list[r - 1][c - 1]

    def __setitem__(self, index, value):
        is_tuple = isinstance(index, tuple)
        length = 1 if not is_tuple else len(index)

        r, c = 0, 0
        if length == 1:  # Setting a Row
            if is_tuple:
                index = index[0]
            r = index

        elif length == 2:  # Setting a Row, a Column, or an Element
            r, c = index

        if c <= 0:  # Setting a Row
            for i, e in enumerate(value):
                self._list[r - 1][i] = e
        elif r <= 0:  # Setting a Column
            for i, e in enumerate(value):
                self._list[i][c - 1] = e
        else:
            self._list[r - 1][c - 1] = value

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Matrix([row * other for row in self._list])

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Matrix([row / other for row in self._list])

    def __add__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Matrix([row + other for row in self._list])

    def __sub__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Matrix([row - other for row in self._list])

def main():
    # size = int(input("Input size:"))
    size = 3

    # matrix = Matrix([[randint(-10, 10) for x in range(size)] for y in range(size)])
    matrix = Matrix(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    )
    print(matrix)
    # print(f"C1: {matrix[0, 1]}")
    matrix[0, 1] *= 2
    # print(f"C1: {matrix[0, 1]}")
    # print(f"R2: {matrix[2, 0]}")
    print(matrix)

    # print("\n".join([" ".join([str(e) for e in row]) for row in matrix]), "\n")

    # start_time = time.time()
    # det = det_laplace(matrix)
    # delta_time2 = time.time() - start_time
    #
    # print(f"determinant = {det} in {delta_time2}s")


if __name__ == '__main__':
    main()
