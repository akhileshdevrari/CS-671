import numpy as np

class model():
    def __init__(self,data,data_label,number_of_class,alpha=0.1,optimizer='grad_desc',beta1=0.9,beta2=0.999,accuracy=0.99,max_iteration=100):
        self.layers = []
        self.alpha = alpha
        self.accuracy = accuracy
        self.data = data
        self.data_label = data_label 
        self.max_iteration = max_iteration
        self.optimizer = optimizer
        self.current_acc = 0
        self.desired_acc = accuracy
        self.number_of_class = number_of_class
        self.cm = np.zeros((self.number_of_class,self.number_of_class),dtype=int)
        self.eplison = 10**(-4)
        self.loss_history = []
        self.accuracy_history = []
        self.beta1 = beta1 
        self.beta2 = beta2
        self.iteration = 0
        
    def add(self,layer):
        """
        Adds layer to the Neural Network model.

        Parameters
        ----------
        layer : layer object
            Individual layer.

        """
        self.layers.append(layer)
    
    def train(self):
        """
        Training Model with initialized hyper-parameters.

        """
        self.iteration = 0
        while((self.iteration < self.max_iteration) and (self.current_acc < self.desired_acc)):
            print self.iteration
            self.cm = np.zeros((self.number_of_class,self.number_of_class),dtype=int)
            self.__do_epoch()
            self.__calculate_accuracy()
            self.accuracy_history.append(self.current_acc)
            print self.current_acc
            
            self.iteration+=1
    
    def predict(self,data,data_label):
        """
        Predicts data_labels given input data. 

        Parameters
        ----------
        data : ndarray
            Input vector representing the data.
        data_label : ndarray
            Input vector representing the associated data labels.

        """
        self.cm = np.zeros((self.number_of_class,self.number_of_class),dtype=int)
        for j in range(len(data)):
            y_label = data_label[j]
            x = data[j]
            y = self.__forprop(x)
            if(self.number_of_class > 2):
                self.cm[y_label.argmax()][y.argmax()] += 1
            else:
                if(y>0.5):
                    self.cm[y_label][1] += 1
                else:
                    self.cm[y_label][0] += 1
        self.__calculate_accuracy()
        print self.current_acc

    def __calculate_accuracy(self):
        """
        Calculates accuracy from confusion matrix.

        """
        self.current_acc = np.trace(self.cm)/(1.0*np.sum(self.cm))

    def __update(self):
        """
        Computes changes in weights, biases in accordance to Optimizer used.

        """
        for layer in self.layers:
#             if(self.optimizer == 'ADAM'):
                
            if(self.optimizer == 'grad_desc_mom'):
                layer.vel1 = self.beta1 * layer.vel1 + self.alpha * layer.dw
                layer.vel2 = self.beta1 * layer.vel2 + self.alpha * layer.db
                layer.update(layer.vel1,layer.vel2)
                
            elif(self.optimizer == 'Adagrad'):
                layer.sim1 += np.square(layer.dw)
                layer.sim2 += np.square(layer.db)
                alpha1 = (self.alpha*layer.dw)/(np.sqrt(layer.sim1+self.eplison))
                alpha2 = (self.alpha*layer.db)/(np.sqrt(layer.sim2+self.eplison))
                layer.update(alpha1,alpha2)
                
            elif(self.optimizer == 'RMS'):
                layer.sim1 = (self.beta1 * layer.sim1) + ((1-self.beta1)*np.square(layer.dw))
                layer.sim2 = (self.beta1 * layer.sim2) + ((1-self.beta1)*np.square(layer.db))
                alpha1 = (self.alpha*layer.dw)/(np.sqrt(layer.sim1+self.eplison))
                alpha2 = (self.alpha*layer.db)/(np.sqrt(layer.sim2+self.eplison))
                layer.update(alpha1,alpha2)
                
            elif(self.optimizer == 'Adam'):
                layer.vel1 = (self.beta1 * layer.vel1) + ((1-self.beta1)*layer.dw)
                layer.vel2 = (self.beta1 * layer.vel2) + ((1-self.beta1)*layer.db)
                layer.sim1 = (self.beta2 * layer.sim1) + ((1-self.beta2)*np.square(layer.dw))
                layer.sim2 = (self.beta2 * layer.sim2) + ((1-self.beta2)*np.square(layer.db))
                corrected_vel1 = layer.vel1/(np.power(1-self.beta1,2))
                corrected_vel2 = layer.vel2/(np.power(1-self.beta1,2))
                corrected_sim1 = layer.sim1/(np.power(1-self.beta2,2))
                corrected_sim2 = layer.sim2/(np.power(1-self.beta2,2))
                alpha1 = (self.alpha*corrected_vel1)/(np.sqrt(corrected_sim1+self.eplison))
                alpha2 = (self.alpha*corrected_vel2)/(np.sqrt(corrected_sim2+self.eplison))
                layer.update(alpha1,alpha2)
            
            else:
                alpha1 = layer.dw * self.alpha 
                alpha2 = layer.db * self.alpha
                layer.update(alpha1,alpha2)
       
    def __get_cost(self,y,y_label):
        """
        Computes loss given predicted data label and actual data label.

        Parameters
        ----------
        y : ndarray
            Input vector representing the predicted data label.
        y_label : ndarray
            Input vector representing the actual data labels.

        """
        if(self.number_of_class > 2):
            y_label = y_label.reshape((y_label.shape[0],1))
            index = y_label.argmax()
            cost = -1*np.log(y[index] + self.eplison)
            return (y_label - y),cost
        else:
            if (y_label == 1):
                return -1*np.log(y + self.eplison),-1*np.log(1 - y + self.eplison)
            else:
                return -1*np.log(1 - y + self.eplison),-1*np.log(1 - y + self.eplison)
                    
    def __do_epoch(self):
        """
        Single pass of entire training data through neural network.

        """
        total_cost = 0
        for j in range(len(self.data)):
            y_label = self.data_label[j]
            x = self.data[j]
            y = self.__forprop(x)
            loss,cost = self.__get_cost(y,y_label)
            total_cost += cost[0]
            if(self.number_of_class > 2):
                self.cm[y_label.argmax()][y.argmax()] += 1
            else:
                if(y>0.5):
                    self.cm[y_label][1] += 1
                else:
                    self.cm[y_label][0] += 1
            self.__backprop(loss)
            self.__update()
        total_cost/=len(self.data)
        self.loss_history.append(total_cost)
            
    def __forprop(self,sample):
        """
        Computes Forward propogation step for all layers.

        Parameters
        ----------
        sample : ndarray
            Input vector for the data.

        Returns
        -------
        ndarray
            Output vector representing the predicted data label.

        """
        for layer in self.layers:
            sample = layer.forward(sample)
        return sample       
    
    def __backprop(self,loss):
        """
        Computes Backward propogation step for all layers.

        Parameters
        ----------
        loss : ndarray
            Input vector for the loss obtained during the forward propogation.

        """
        sample = loss
        for i in range(len(self.layers)-1,-1,-1):
            sample = self.layers[i].backward(sample)
        
