import networkx as NX
import pygraphviz as PG

G = PG.AGraph()
nlist = "A B C D E".split()
a, b = "A A B", "B C D"
elist = zip(a.split(), b.split())

G.add_nodes_from(nlist)
G.add_edges_from(elist)
G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="blue", len="2.0", width="2.0")

print(G.edge_attr)
# returns {'color': 'red', 'width': '', 'len': '2.0'}

# add new edge with custom length (all others have length=2.0):
G.add_edge("C", "E", len="3.0", color="blue", width="2.0")

G.add_edge("A", "F", len="1.0", color="red", width="2.0")
edge = G.get_edge("C", "E")
print(G.edge_attr)
# returns {'color': 'blue', 'width': '2.0', 'len': '3.0'}

# and you can confirm that introspection by drawing & printing this graph:
G.draw('somefolderandfilename.png', format='png', prog='neato')