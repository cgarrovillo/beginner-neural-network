import numpy as np
from nn import NeuralNetwork

# Sample dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train the neural network
network = NeuralNetwork()
network.train(data, all_y_trues)

# Vectors based off of weight -135lbs and height -66in
emily = np.array([-7, -3])                          # 128 pounds, 63 inches
frank = np.array([20, 2])                           # 155 pounds, 68 inches

# UFC fighter data
jonjones = np.array([70, 10])   # from EA
dc = np.array([70, 5])          # from EA
manderson = np.array([12, 7])
nunes = np.array([0, 2])

christian = np.array([-15, -2])

# Calculate 
result1 = network.feedforward(emily)
result2 = network.feedforward(frank)

result3 = network.feedforward(jonjones)
result4 = network.feedforward(dc)
result5 = network.feedforward(manderson)
result6 = network.feedforward(nunes)

result7 = network.feedforward(christian)



# print("Emily (dataset): %.3f" % result1)   # 0.951 - F
# print("Frank (dataset): %.3f" % result2)   # 0.039 - M

def gender(result):
    return "Female" if result > 0.5 else "Male"

print("Emily (dataset) is a %s" % gender(result1))
print("Frank (dataset) is a %s" % gender(result2))
print("Jon Jones (EA Sports) is a %s" % gender(result3))
print("DC (EA Sports) is a %s" % gender(result4))
print("Manderson (Wikipedia) is a %s" % gender(result5))
print("Nunes (Wikipedia) is a %s" % gender(result6))
print("Therefore, Christian (input) is a %s" % gender(result7))
