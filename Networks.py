from typing import List


class Node:
    def __init__(self, ip, port, targets:List["Node"]) -> None:
        self.ip = ip
        self.port = port
        self.targets = targets
    
    def get_ip(self):
        return self.ip
    
    def get_port(self):
        return self.port
    
    def get_targets(self):
        return self.targets
    
    def find_target_by_ip_port(self, ip, port):
        for t in self.targets:
            if t.get_ip() == ip and t.get_port == port:
                return t
        return -1
        
        
        
class Edge:
    def __init__(self, source:Node, target:Node) -> None:
        self.source = source
        self.target = target
        
        
        
        
        
class Network:
    def __init__(self, nodes:List[Node], edges:List[Edge]) -> None:
        self.nodes = nodes
        self.edges = edges
    
    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        return self.edges