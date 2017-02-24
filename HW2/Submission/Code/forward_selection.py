
import Queue
import copy


class ForwardSelection():

    def __init__(self, training_data):
        self.training_data = training_data
        self.best_params = []
        self.current_params = []
        self.best_score = -float('inf')
        self.betterized = True
        self.heap = []


    def set_score(self, score):
        if score > self.best_score:
            self.best_params = self.current_params
            self.best_score = score
            self.betterized = True

    def get_new_heap(self):
        if (len(self.heap) == 0) and not self.betterized:
            return  self.betterized
        elif(len(self.best_params) == self.training_data.shape[1]):
            return False
        else:
            if len(self.heap) == 0:
                self.set_new_heap()
            return self.heap.pop()

    def set_new_heap(self):
        self.betterized = False

        for i in range(0,self.training_data.shape[1]):
            if i not in self.best_params:
                elem = copy.copy(self.best_params)
                elem.append(i)
                self.heap.append(elem)

    def transform(self, data = None):
        if data == None:
            data = self.training_data
        popped = self.get_new_heap()
        if popped is False:
            return popped
        else:
            return data[:, popped]


# train_x, train_y, test_x = pipe.get_data('AirFoil')
# #print(train_x.shape)
# test = ForwardSelection(train_x)
# while True:
#     test_array = test.transform()
#     if test_array is False:
#         break
#




