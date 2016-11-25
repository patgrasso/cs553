

from nodes import Node, PartitionNode, AddNode

stream = Node() \
    .then(AddNode(10)) \
    .then(PartitionNode(.6, .2, .2))

print(stream(list(range(10))))
