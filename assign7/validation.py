# Patrick Grasso
# PageRank algorithm

# T_j   = docs linking to page A
# C(T)  = links in page T
# PR(A) = d + (1-d)*sum(j; PR(T_j) / C(T_j))

iters   = 25

#           A  B  C
graph   = [[0, 1, 1], # A
           [1, 0, 0], # B
           [1, 0, 0]] # C

ranks   =  [1, 1, 1]

d = 0.1

def prnt(i, arr):
    print(("{:2}, " + ", ".join(["{:.2f}"]*len(ranks))).format(i, *arr))

def PR(page):
    term = lambda other: ranks[other] * graph[page][other] / sum(graph[other])
    return d + (1-d)*sum(map(term, range(len(ranks))))

for i in range(iters + 1):
    prnt(i, ranks)
    for page in range(len(ranks)):
        ranks[page] = PR(page)

