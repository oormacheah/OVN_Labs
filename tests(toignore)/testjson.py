class Triangle:
    def __init__(self, a, b, c):
        self._x = a
        self._y = b
        self._z = c
    @property
    def a(self):
        return self._x
