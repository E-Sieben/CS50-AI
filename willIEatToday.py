import numpy as np

# Simple attempt to import pomegranate, fall back to custom implementation if not available
try:
    from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, Node, BayesianNetwork
    print("Using pomegranate library")
except ImportError:
    print("Using fallback implementation without pomegranate")
    
    # Minimal classes for fallback implementation
    class DiscreteDistribution:
        def __init__(self, probabilities):
            self.probabilities = probabilities
            
    class Node:
        def __init__(self, distribution, name=None):
            self.distribution = distribution
            self.name = name
            self.parameters = [distribution.probabilities] if hasattr(distribution, 'probabilities') else [{}]
            
    class ConditionalProbabilityTable:
        def __init__(self, table, parents):
            self.table = {}
            for row in table:
                outcome, probability = row[-2], row[-1]
                conditions = tuple(row[:-2])
                if conditions not in self.table:
                    self.table[conditions] = {}
                self.table[conditions][outcome] = probability
            self.parents = parents
            self.parent_names = []
            
    class BayesianNetwork:
        def __init__(self, name=None):
            self.name = name
            self.states = []
            self.edges = []
            
        def add_states(self, *states):
            self.states.extend(states)
            
        def add_edge(self, parent, child):
            self.edges.append((parent, child))
            if hasattr(child.distribution, 'parent_names'):
                child.distribution.parent_names.append(parent.name)
            
        def bake(self):
            pass
            
        def predict_proba(self, evidence):
            results = []
            for node in self.states:
                if node.name == "eat":
                    if hasattr(node.distribution, 'table'):
                        conditions = tuple([evidence.get(parent_name) for parent_name in node.distribution.parent_names])
                        if conditions in node.distribution.table:
                            node.parameters[0] = node.distribution.table[conditions]
                        else:
                            node.parameters[0] = {"yes": 0.5, "no": 0.5}
                results.append(node)
            return results

def create_bayesian_network():
    # Node 1: Dead status
    dead = DiscreteDistribution({
        "yes": 0.01,
        "maybe": 0.04,
        "no": 0.95
    })
    
    # Node 2: Money status
    money = DiscreteDistribution({
        "no": 0.3,
        "a little bit": 0.5,
        "yes": 0.2
    })
    
    # Node 3: Hunger status
    hungry = DiscreteDistribution({
        "no": 0.1,
        "a little bit": 0.2,
        "maybe": 0.3,
        "sure": 0.3,
        "extremely": 0.1
    })
    
    # Node 4: Will Eat Today (depends on Dead, Money, and Hungry)
    eat_cpt = ConditionalProbabilityTable([
        # If dead, almost certainly won't eat
        ["yes", "no", "no", "no", 0.99],
        ["yes", "no", "no", "yes", 0.01],
        ["yes", "no", "a little bit", "no", 0.99],
        ["yes", "no", "a little bit", "yes", 0.01],
        ["yes", "no", "maybe", "no", 0.99],
        ["yes", "no", "maybe", "yes", 0.01],
        ["yes", "no", "sure", "no", 0.99],
        ["yes", "no", "sure", "yes", 0.01],
        ["yes", "no", "extremely", "no", 0.99],
        ["yes", "no", "extremely", "yes", 0.01],
        
        ["yes", "a little bit", "no", "no", 0.99],
        ["yes", "a little bit", "no", "yes", 0.01],
        ["yes", "a little bit", "a little bit", "no", 0.99],
        ["yes", "a little bit", "a little bit", "yes", 0.01],
        ["yes", "a little bit", "maybe", "no", 0.99],
        ["yes", "a little bit", "maybe", "yes", 0.01],
        ["yes", "a little bit", "sure", "no", 0.99],
        ["yes", "a little bit", "sure", "yes", 0.01],
        ["yes", "a little bit", "extremely", "no", 0.99],
        ["yes", "a little bit", "extremely", "yes", 0.01],
        
        ["yes", "yes", "no", "no", 0.99],
        ["yes", "yes", "no", "yes", 0.01],
        ["yes", "yes", "a little bit", "no", 0.99],
        ["yes", "yes", "a little bit", "yes", 0.01],
        ["yes", "yes", "maybe", "no", 0.99],
        ["yes", "yes", "maybe", "yes", 0.01],
        ["yes", "yes", "sure", "no", 0.99],
        ["yes", "yes", "sure", "yes", 0.01],
        ["yes", "yes", "extremely", "no", 0.99],
        ["yes", "yes", "extremely", "yes", 0.01],
        
        # If maybe dead, might eat depending on other factors
        ["maybe", "no", "no", "no", 0.9],
        ["maybe", "no", "no", "yes", 0.1],
        ["maybe", "no", "a little bit", "no", 0.8],
        ["maybe", "no", "a little bit", "yes", 0.2],
        ["maybe", "no", "maybe", "no", 0.7],
        ["maybe", "no", "maybe", "yes", 0.3],
        ["maybe", "no", "sure", "no", 0.6],
        ["maybe", "no", "sure", "yes", 0.4],
        ["maybe", "no", "extremely", "no", 0.5],
        ["maybe", "no", "extremely", "yes", 0.5],
        
        ["maybe", "a little bit", "no", "no", 0.8],
        ["maybe", "a little bit", "no", "yes", 0.2],
        ["maybe", "a little bit", "a little bit", "no", 0.6],
        ["maybe", "a little bit", "a little bit", "yes", 0.4],
        ["maybe", "a little bit", "maybe", "no", 0.5],
        ["maybe", "a little bit", "maybe", "yes", 0.5],
        ["maybe", "a little bit", "sure", "no", 0.3],
        ["maybe", "a little bit", "sure", "yes", 0.7],
        ["maybe", "a little bit", "extremely", "no", 0.2],
        ["maybe", "a little bit", "extremely", "yes", 0.8],
        
        ["maybe", "yes", "no", "no", 0.7],
        ["maybe", "yes", "no", "yes", 0.3],
        ["maybe", "yes", "a little bit", "no", 0.5],
        ["maybe", "yes", "a little bit", "yes", 0.5],
        ["maybe", "yes", "maybe", "no", 0.3],
        ["maybe", "yes", "maybe", "yes", 0.7],
        ["maybe", "yes", "sure", "no", 0.2],
        ["maybe", "yes", "sure", "yes", 0.8],
        ["maybe", "yes", "extremely", "no", 0.1],
        ["maybe", "yes", "extremely", "yes", 0.9],
        
        # If not dead, eating depends on money and hunger
        ["no", "no", "no", "no", 0.9],
        ["no", "no", "no", "yes", 0.1],
        ["no", "no", "a little bit", "no", 0.8],
        ["no", "no", "a little bit", "yes", 0.2],
        ["no", "no", "maybe", "no", 0.6],
        ["no", "no", "maybe", "yes", 0.4],
        ["no", "no", "sure", "no", 0.4],
        ["no", "no", "sure", "yes", 0.6],
        ["no", "no", "extremely", "no", 0.3],
        ["no", "no", "extremely", "yes", 0.7],
        
        ["no", "a little bit", "no", "no", 0.7],
        ["no", "a little bit", "no", "yes", 0.3],
        ["no", "a little bit", "a little bit", "no", 0.4],
        ["no", "a little bit", "a little bit", "yes", 0.6],
        ["no", "a little bit", "maybe", "no", 0.3],
        ["no", "a little bit", "maybe", "yes", 0.7],
        ["no", "a little bit", "sure", "no", 0.2],
        ["no", "a little bit", "sure", "yes", 0.8],
        ["no", "a little bit", "extremely", "no", 0.1],
        ["no", "a little bit", "extremely", "yes", 0.9],
        
        ["no", "yes", "no", "no", 0.5],
        ["no", "yes", "no", "yes", 0.5],
        ["no", "yes", "a little bit", "no", 0.3],
        ["no", "yes", "a little bit", "yes", 0.7],
        ["no", "yes", "maybe", "no", 0.2],
        ["no", "yes", "maybe", "yes", 0.8],
        ["no", "yes", "sure", "no", 0.1],
        ["no", "yes", "sure", "yes", 0.9],
        ["no", "yes", "extremely", "no", 0.05],
        ["no", "yes", "extremely", "yes", 0.95],
    ], [dead, money, hungry])
    
    # Create nodes and network
    d = Node(dead, name="dead")
    m = Node(money, name="money")
    h = Node(hungry, name="hungry")
    e = Node(eat_cpt, name="eat")
    
    model = BayesianNetwork("Will I Eat Today")
    model.add_states(d, m, h, e)
    
    model.add_edge(d, e)
    model.add_edge(m, e)
    model.add_edge(h, e)
    
    model.bake()
    return model

def query_network(model, dead_status, money_status, hungry_status):
    evidence = {"dead": dead_status, "money": money_status, "hungry": hungry_status}
    beliefs = model.predict_proba(evidence)
    
    for node in beliefs:
        if node.name == "eat":
            return node.parameters[0]

def main():
    print("Will I Eat Today? - Bayesian Network Decision System")
    print("--------------------------------------------------")
    
    model = create_bayesian_network()
    
    # Get user inputs
    print("\nPlease answer the following questions:")
    
    print("\nAm I dead?")
    print("1. Yes\n2. Maybe\n3. No")
    dead_status = ["yes", "maybe", "no"][int(input("Enter choice (1-3): ")) - 1]
    
    print("\nDo I have money?")
    print("1. No\n2. A little bit\n3. Yes")
    money_status = ["no", "a little bit", "yes"][int(input("Enter choice (1-3): ")) - 1]
    
    print("\nAm I hungry?")
    print("1. No\n2. A little bit\n3. Maybe\n4. Sure\n5. Extremely")
    hungry_status = ["no", "a little bit", "maybe", "sure", "extremely"][int(input("Enter choice (1-5): ")) - 1]
    
    eat_probs = query_network(model, dead_status, money_status, hungry_status)
    
    print("\nResults:")
    print(f"Probability of eating today: {eat_probs['yes']:.2%}")
    print(f"Probability of not eating today: {eat_probs['no']:.2%}")
    
    if eat_probs['yes'] > eat_probs['no']:
        print("\nConclusion: You will most likely eat today.")
    else:
        print("\nConclusion: You will most likely NOT eat today.")

if __name__ == "__main__":
    main()
