import math


def dot(u, v):
    assert len(u) == len(v)
    _sum = 0
    for x, y in zip(u, v):
        _sum += x * y
    return _sum

class Vector:

    INNER_PRODUCT = dot

    def __init__(self, *args):
        self._list = [arg for arg in args]

    def __getitem__(self, index):
        return self._list[index - 1]

    def __iter__(self):
        return iter([self[i + 1] for i in range(len(self))])

    def __len__(self):
        return len(self._list)

    def __add__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, Vector)
        if isinstance(other, Vector):
            return Vector(*[self[i + 1] + other[i + 1] for i in range(min(len(self), len(other)))])
        return Vector(*[row + other for row in self])

    __iadd__ = __add__

    def __sub__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, Vector)
        if isinstance(other, Vector):
            return Vector(*[self[i + 1] - other[i + 1] for i in range(min(len(self), len(other)))])
        return Vector(*[row - other for row in self])

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, Vector)

        if isinstance(other, Vector):
            if not len(self) == len(other):
                raise IndexError
            return sum([self[i + 1] * other[i + 1] for i in range(len(self))])
        return Vector(*[row * other for row in self])

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return Vector(*[row / other for row in self])

    def __str__(self):
        return "<" + ", ".join([str(e) for e in self]) + ">"

    def append(self, element):
        self._list.append(element)

    def magnitude(self):
        return math.sqrt(sum([e*e for e in self]))

    def normalized(self):
        return self / self.magnitude()

    def adjoin(self, other):
        assert isinstance(other, Vector)
        return Vector(*[self[i + 1] if i < len(self) else other[i - len(self) + 1] for i in range(len(self) + len(other))])

    def inner_product(self, other):
        return Vector.INNER_PRODUCT(self, other)

    def project(self, other):
        return self * (self.inner_product(other) / self.inner_product(self))


class Matrix:
    def __init__(self, matrix):
        self._list = []
        if len(matrix) == 0:
            return
        if isinstance(matrix[0], list) or isinstance(matrix[0], tuple) or isinstance(matrix[0], Vector):
            for row in matrix:
                m_row = []
                for element in row:
                    m_row.append(element)
                self._list.append(m_row)
        else:
            self._list.append([element for element in matrix])

        self.factors = 1

    def __str__(self):
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
            if (i + 1) % self.columns == 0:
                string += "\n"

        return string[:-1]

    def __iter__(self):
        return iter([self[i + 1, 0] for i in range(len(self))])

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        # print(index)
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

    def __add__(self, other):
        assert isinstance(other, float) or isinstance(other, int) \
               or (isinstance(other, Matrix) and self.rows == other.rows and self.columns == other.columns)

        if isinstance(other, float) or isinstance(other, int):
            return Matrix([row + other for row in self])
        elif isinstance(other, Matrix):
            return Matrix([[self[r + 1, c + 1] + other[r + 1, c + 1] for c in range(self.columns)] for r in range(self.rows)])

    __iadd__ = __add__

    def __sub__(self, other):
        assert isinstance(other, float) or isinstance(other, int) \
               or (isinstance(other, Matrix) and self.rows == other.rows and self.columns == other.columns)
        if isinstance(other, float) or isinstance(other, int):
            return Matrix([row - other for row in self])
        elif isinstance(other, Matrix):
            return Matrix([[self[r + 1, c + 1] - other[r + 1, c + 1] for c in range(self.columns)] for r in range(self.rows)])

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, Matrix)

        if isinstance(other, float) or isinstance(other, int):
            return Matrix([row * other for row in self])
        elif isinstance(other, Matrix):
            if not self.columns == other.rows:
                raise IndexError
            return Matrix([[self[r + 1, 0] * other[0, c + 1] for c in range(other.columns)] for r in range(self.rows)])

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)

        if isinstance(other, float) or isinstance(other, int):
            return Matrix([self[row + 1, 0] / other for row in range(self.rows)])

    def transpose(self):
        return Matrix([self[0, i + 1] for i in range(self.columns)])

    def determinate(self):
        return self.rref().factors

    def adjoin(self, other):
        assert isinstance(other, Matrix) and self.rows == other.rows
        return Matrix([self[i + 1, 0].adjoin(other[i + 1, 0]) for i in range(self.rows)])

    def adjoin_vertical(self, other):
        assert isinstance(other, Matrix) and self.columns == other.columns
        return Matrix([self[0, i + 1].adjoin(other[0, i + 1]) for i in range(self.columns)]).transpose()

    def append_row(self, row):
        assert isinstance(row, Vector) and self.columns == len(row)
        self._list.append(list(row))

    def append_column(self, column):
        assert isinstance(column, Vector) and self.rows == len(column)
        for i in range(len(column)):
            self._list[i].append(column[i + 1])

    def sub(self, row, column):
        assert 1 <= row <= self.rows and 1 <= column <= self.columns
        return Matrix([[self[r + (r >= row), c + (c >= column)] for r in range(1, self.rows)] for c in range(1, self.columns)])

    def adjugate(self):
        return Matrix([[self.sub(r + 1, c + 1).det * (-1) ** ((r + c) % 2) for r in range(self.rows)] for c in range(self.columns)]).transpose()

    def inverse(self):
        return self.adjugate() / self.det

    def _row_count(self):
        return len(self._list)

    def _column_count(self):
        return len(self._list[0])

    def ref(self):
        matrix = Matrix(self)
        for i in range(matrix.rows):
            if i >= matrix.columns:
                return matrix

            if matrix[i + 1, i + 1] == 0:  # Runs this if there is a 0 where a number is needed
                for j in range(i + 1, matrix.rows):
                    if matrix[j + 1, i + 1] != 0:
                        matrix[i + 1, 0], matrix[j + 1, 0] = matrix[j + 1, 0], matrix[i + 1, 0]
                        break
            matrix.factors *= matrix[i + 1, i + 1]
            matrix[i + 1, 0] /= matrix[i + 1, i + 1]
            for j in range(i + 1, matrix.rows):
                matrix[j + 1, 0] -= matrix[i + 1, 0] * matrix[j + 1, i + 1]
        return matrix

    def rref(self):
        matrix = self.ref()
        for i in range(1, matrix.rows):
            if i >= matrix.columns:
                return
            for j in range(0, i):
                matrix[j + 1, 0] -= matrix[i + 1, 0] * matrix[j + 1, i + 1]
        return matrix

    rows = property(_row_count)
    columns = property(_column_count)
    det = property(determinate)

def I(size: int):
    return Matrix([[int(row == column) for row in range(size)] for column in range(size)])

def det_laplace(matrix: Matrix):
    assert matrix.rows == matrix.columns
    if matrix.rows == 2:
        return matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]

    return sum([matrix[1, c + 1] * det_laplace(matrix.sub(1, c + 1)) * ((-1) ** (c % 2)) for c in range(matrix.columns)])

def cramers(a: Matrix, b: Matrix):
    assert (det := a.det) != 0 and a.rows == b.rows
    return [Matrix([b[0, 1] if c == c2 else a[0, c + 1] for c in range(a.columns)]).transpose().det/det for c2 in range(a.columns)]

def orthonormalization_gram_schmidt(basis):
    orthonormalization = Vector()
    for i in range(len(basis)):
        vector = basis[i]
        for j in range(i - 1, 0, -1):
            vector -= basis[j].project(basis[i])

        orthonormalization.append(vector.normalized())
    return orthonormalization

def main():
    A = Matrix([
        [1, 0, 1],
        [1, -1, 0],
        [1, 2, 1]
    ])
    b = Matrix([-2, 1, 0]).transpose()
    print(cramers(A, b))



if __name__ == '__main__':
    main()
