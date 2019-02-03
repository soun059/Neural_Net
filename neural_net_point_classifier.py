import numpy as np
import matplotlib.pyplot as plt

def create_dataset():
    data = {}
    blue_data = []
    red_data = []
    points1 = list(np.arange(0,5.1,0.1))
    points2 = list(np.arange(5,10.1,0.1))
    for i in range(100):
        idx = np.random.randint(0,len(points1)-1)
        idx2 = np.random.randint(0,len(points2)-1)
        red_data.append([points1[idx],points1[idx2]])
        blue_data.append([points1[idx],points2[idx2]])
        red_data.append([points2[idx],points2[idx2]])
        blue_data.append([points2[idx],points1[idx2]])
    data = { 'red':red_data ,'blue':blue_data  }
    return data

class neural_net:

    def __init__(self):

        self.inner_layer_weights = [[i for i in range(2)] for j in range(4)]
        self.inner_layer_bias = [i for i in range(4)]
        self.hidden_layer_weights = [[k for k in range(4)] for l in range(2)]
        self.hidden_layer_bias = [m for m in range(2)]
        self.data = []
        self.hidden_layer = []
        self.output_layer = []
        self.hidden_layer_error = []
        self.output_layer_error = []
        
    def fit(self,data):

        red = data['red']
        blue = data['blue']
        total_data = []
        for i in range(2*len(red)):
            if i>2:
                n = np.random.randint(0,2)
                if n == 0:
                    self.data.append([blue[int(i/2)],[0,1]])
                else:
                    self.data.append([red[int(i/2)],[1,0]])
            else:
                n = np.random.randint(0,2)
                if n == 0:
                    self.data.append([blue[i],[0,1]])
                else:
                    self.data.append([red[i],[1,0]])
                    
    def process(self,epoch,learning_rate):
        self.learning_rate = learning_rate

        for i in range(epoch):
            for j in self.data:
                self.feed_forward(j[0])
                self.error_cal(j[1])
                self.back_prop(j[0])

    def feed_forward(self,data):

        del self.hidden_layer[:]
        del self.output_layer[:]

        for i in range(len(self.inner_layer_bias)):
            a = np.dot(data,np.transpose(self.inner_layer_weights[i])) + self.inner_layer_bias[i]
            self.hidden_layer.append(max(0,a))
        for j in range(len(self.hidden_layer_bias)):
            a = np.dot(self.hidden_layer,np.transpose(self.hidden_layer_weights[j])) + self.hidden_layer_bias[j]
            self.output_layer.append(max(0,a))

    def error_cal(self,data):

        del self.output_layer_error[:]
        del self.hidden_layer_error[:]

        self.output_layer_error = list(np.subtract(data,self.output_layer))
        for i in range(len(self.inner_layer_weights)):
            self.hidden_layer_error.append(np.dot(self.output_layer_error,np.transpose(np.transpose(self.hidden_layer_weights)[i])))

    def back_prop(self,data):
        
        for i in range(len(self.hidden_layer_weights)):
            for j in range(len(self.hidden_layer_weights[i])):
                self.hidden_layer_weights[i][j] += self.learning_rate*self.hidden_layer[j]*self.output_layer_error[i]
            #self.hidden_layer_bias[i] += self.learning_rate*self.output_layer_error[i]

        for i in range(len(self.inner_layer_weights)):
            for j in range(len(self.inner_layer_weights[i])):
                self.inner_layer_weights[i][j] += self.learning_rate*self.hidden_layer_error[i]*data[j]
            #self.inner_layer_bias[i] += self.learning_rate*self.hidden_layer_error[i]

    def prediction(self,x,y):
        data = [x,y]
        self.feed_forward(data)
        plt.scatter(x,y,c='g')
        print(self.output_layer)
        if(self.output_layer[0]<self.output_layer[1])
                print("The point is blue")
        else
                print("The point is red")
                                     




data = create_dataset()
for i in data:
    for j in data[i]:
        if i=='red':
            plt.scatter(j[0],j[1],c= 'r')
        else:
            plt.scatter(j[0],j[1],c= 'b')
fnn = neural_net()
fnn.fit(data)
fnn.process(400,0.001) #use plt.show() on terminal of idle or any other python-based terminal to get the plot of graphics data on which classification is done.
                        #use fnn.predict to determine the prediction of the data , like put fnn.predict(1,2) 


#p = ax.pcolor(X, Y/(2*np.pi), Z, cmap=matplotlib.cm.RdBu
