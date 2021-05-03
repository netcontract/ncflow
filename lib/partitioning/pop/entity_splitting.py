import numpy as np
import heapq
import copy

class MaxHeapObj(object):

    def __init__(self, entity):
        self.entity = entity
    
    # reverse comparison of demands, since heapq implemented as minheap
    def __lt__(self, other):
        return self.entity[-1] > other.entity[-1]

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.entity)

    def get_entity(self):
        return self.entity

    def split(self, factor):
        self.entity[-1] = self.entity[-1]*factor
        

def halve(entity_mho):
    halved_entities_mho = [MaxHeapObj(copy.deepcopy(entity_mho.get_entity())) for _ in range(2)]
    for entity_mho in halved_entities_mho:
        entity_mho.split(1/2.0)

    return halved_entities_mho

# split the max entities in half until add_fraction new entities are formed
def split_entities(entity_list, add_fraction):

    print("splitting for additional " + str(add_fraction) + " entities")

    num_entities = len(entity_list)
    num_new_entities = np.round(num_entities*add_fraction)

    # creat MaxHeapObject list of entities    
    entity_mho_list = [MaxHeapObj(entity) for entity in entity_list]

    # create heap of maxHeapObjects (a maxheap)
    heapq.heapify(entity_mho_list)

    while len(entity_mho_list) < num_entities + num_new_entities:
        largest_entity_mho = heapq.heappop(entity_mho_list)

        # split largest entity
        new_entities = halve(largest_entity_mho)

        # add it to heap
        for new_entity in new_entities:
            heapq.heappush(entity_mho_list, new_entity)

    resulting_entity_list = [entity_mho.get_entity() for entity_mho in entity_mho_list]

    return resulting_entity_list
