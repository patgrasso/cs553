

class Node:

    def __init__(self):
        self.next = None

    def then(self, node):
        if self.next != None:
            self.next.then(node)
        elif isinstance(node, Node):
            self.next = node
        else:
            raise TypeError(str(node) + " is not a Node")
        return self

    def __call__(self, value):
        return self.next(value) if self.next != None else None


class AddNode(Node):

    def __init__(self, toAdd=0):
        super().__init__()
        self.toAdd = toAdd

    def __call__(self, value):
        if isinstance(value, list):
            value = list(map(lambda x: x + self.toAdd, value))
        else:
            value += self.toAdd
        return value if self.next == None else self.next(value)


