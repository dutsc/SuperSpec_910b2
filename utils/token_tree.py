import torch
from .logger import init_logger
logger = init_logger(__name__)

class TokenLogitsPair:
    def __init__(self, token, logits):
        self.token = token
        self.logits = logits
    def to(self, device, non_blocking=False):
        # 将token和logits转移到指定的设备，并更新这些属性
        self.token = self.token.to(device, non_blocking=non_blocking)
        self.logits = self.logits.to(device, non_blocking=non_blocking)
        return self

class TreeNode:
    def __init__(self, token_logits_pair=None, parent=None):
        self.token_logits_pair = token_logits_pair
        self.parent = parent
        self.children = {}
        self.mask = 0   # rollback的时候用

    def add_child(self, token_logits_pair):
        token = token_logits_pair.token.item()
        if token not in self.children:
            self.children[token] = TreeNode(token_logits_pair, self)
        return self.children[token]

    def add_sequence(self, sequence):
        node = self
        for token_logits_pair in sequence:
            node = node.add_child(token_logits_pair)

    
    def print_tree(self, prefix="", is_last=True):
        if self.token_logits_pair:
            token = self.token_logits_pair.token
            if isinstance(token, torch.Tensor):
                if token.numel() == 1:
                    token = str(token.item())
                else:
                    token = str(token.tolist())
        else:
            token = 'Root'
        connector = "└── " if is_last else "├── "
        line = prefix + connector + token
        print(line)
        child_prefix = prefix + ("    " if is_last else "│   ")
        children = list(self.children.values())
        for i, child in enumerate(children):
            is_child_last = i == (len(children) - 1) 
            child.print_tree(child_prefix, is_child_last)
            
    def print_mask(self, prefix="", is_last=True):
        if self.token_logits_pair:
            mask = self.mask
        else:
            mask = 'Root'
        connector = "└── " if is_last else "├── "
        line = prefix + connector + str(mask)
        print(line)
        child_prefix = prefix + ("    " if is_last else "│   ")
        children = list(self.children.values())
        for i, child in enumerate(children):
            is_child_last = i == (len(children) - 1) 
            child.print_mask(child_prefix, is_child_last)


    def dfs(self, visit_func):
        if self.token_logits_pair:
            visit_func(self.token_logits_pair.token)
        for child in self.children.values():
            child.dfs(visit_func)


#---------------------------methods---------------------------------

def create_index_mapping(node, index_mapping, current_index=0):
    # 为每个节点的token创建索引映射
    if node.token_logits_pair:
        # 使用token作为键 改成使用node作为键
        token = node.token_logits_pair.token.item()
        index_mapping[node] = current_index
        current_index += 1
    for child in node.children.values():
        current_index = create_index_mapping(child, index_mapping, current_index)
    return current_index

def print_index_mapping(map):
    for i in list(map.keys()):
        if i.token_logits_pair == None:
            print(f'None:      {map[i]}')
        else:
            print(f'{i.token_logits_pair.token}:    {map[i]}')

def create_sequence_and_mask(root):
    sequence = []  # 用于存储序列
    index_mapping = {}  # 用于映射节点到序列中的索引

    def dfs_sequence(node):
        if node.token_logits_pair:
            token = node.token_logits_pair.token.item()
            sequence.append(token)
            index_mapping[node] = len(sequence) - 1
        for child in node.children.values():
            dfs_sequence(child)

    # 首先，构建序列并映射每个节点到它在序列中的索引
    dfs_sequence(root)

    # 创建一个N*N的全0矩阵，N是序列的长度
    mask = torch.zeros((len(sequence), len(sequence)))

    def dfs_mask(node, visible_indices):
        if node in index_mapping:
            current_index = index_mapping[node]
            # 标记当前节点和它的所有祖先为可见
            for idx in visible_indices:
                mask[current_index, idx] = 1
            mask[current_index, current_index] = 1  # 节点对自己也是可见的
            visible_indices.append(current_index)

        for child in node.children.values():
            dfs_mask(child, visible_indices)
        
        if node in index_mapping:
            visible_indices.pop()  # 返回上一层前，移除当前节点的索引

    # 构建遮罩
    dfs_mask(root, [])
    return sequence, mask


def fine_index_mapping(map):
    nodes = list(map.keys())
    head_nodes = []
    for node in nodes[1:]:
        if map[node] != map[node.parent]+1:  #处理不了一开始就有两个分支的情况，因为map里没有以root(没有存储token)为键的值
            head_nodes.append(node)      #这里先处理成map[root]=-1
    for i in head_nodes:
        map[i] = map[i.parent] + 1
    return map

def create_mask_seq(root):
    seq = []
    def dfs_mask(node):
        if node.token_logits_pair:
            seq.append(node.mask)
        for child in node.children.values():
            dfs_mask(child)
    dfs_mask(root)
    return seq