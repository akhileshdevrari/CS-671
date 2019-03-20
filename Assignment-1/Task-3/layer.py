import numpy as np

class layer():
    def __init__(self,inputs,units,activation='relu',dropout=False,keep_prob=0.8):
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.b = np.zeros((self.units,1))
        self.input_data = 0
        self.z = 0
        self.a = 0
        self.dw = 0
        self.db = 0
        self.da = 0
        self.vel1 = 0
        self.vel2 = 0
        self.sim1 = 0
        self.sim2 = 0
        self.__initialize_weight()
        
        
    def __sigmoid(self, x):
        
        """
        Computes sigmoid activation.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied activation

        """
        return 1 / (1 + np.exp(-x))

    def __initialize_weight(self):

        """
        Initiates weights of the neural layer in accordance to Xavier's Initialization.

        """
        
        if(self.activation == 'relu'):
            self.w = np.random.rand(self.units,self.inputs) * np.sqrt(2.0/self.inputs)
        elif(self.activation == 'tanh'):
            self.w = np.random.rand(self.units,self.inputs) * np.sqrt(1.0/self.inputs)
        else:
            self.w = np.random.rand(self.units,self.inputs) * np.sqrt(2.0/(self.units+self.inputs))
            
    def __sigmoid_d(self, x):

        """
        Computes sigmoid deivative.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied derivative

        """
        
        return x * (1 - x)
    
    def __softmax(self,x):
        """
        Computes softmax activation.
        
        Useful for multi-class classification problems.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied activation

        """
        x = x - np.mean(x)
        expx = np.exp(x)
        return expx / expx.sum()
    
    def __tanh(self, x):
        
        """
        Computes tanh activation.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied activation

        """
        return 1 / (1 + np.exp(-x))

    def __tanh_d(self, x):

        """
        Computes tanh deivative.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied derivative

        """
        return (1 + x) * (1 - x)
    
    def __relu(self,x):
        """
        Computes ReLU activation.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied activation

        """
        return np.maximum(0,x) 
    
    def __relu_d(self,x):

        """
        Computes ReLU deivative.

        Parameters
        ----------
        x : ndarray
            Input vector

        Returns
        -------
        ndarray
            Output vector with applied derivative

        """
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def update(self,alpha1,alpha2):
        """
        Updates weights and biases in accordance to Optimizer.

        Parameters
        ----------
        alpha1 : ndarray
            Input vector for change in weights
        alpha1 : ndarray
            Input vector for change in biases

        """
        self.w += alpha1
        self.b += alpha2
    
    def forward(self,input_data):
        """
        Computes Forward propogation step.

        Parameters
        ----------
        input_data : ndarray
            Input vector for incoming activation terms.

        Returns
        -------
        ndarray
            Output vector for outgoing activation terms.

        """
        input_data = input_data.reshape((input_data.shape[0],1))
        self.input_data = input_data
        self.z = np.dot(self.w,input_data) 
        self.a = self.z
        if(self.activation == 'sigmoid'):
            self.a = self.__sigmoid(self.a)
        elif(self.activation == 'tanh'):
            self.a = self.__tanh(self.a)
        elif(self.activation == 'relu'):
            self.a = self.__relu(self.a)
        elif(self.activation == 'softmax'):
            self.a = self.__softmax(self.a)
        else:
            pass
        if(self.dropout):
            drop = np.random.rand(self.a.shape[0],self.a.shape[1]) < self.keep_prob
            self.a = np.multiply(self.a,drop)
            self.a /= self.keep_prob
        return self.a
            
    def backward(self,error):
        """
        Computes Backward propogation step.

        Parameters
        ----------
        error : ndarray
            Input vector for incoming error terms.

        Returns
        -------
        ndarray
            Output vector for outgoing error terms.

        """
        if(self.activation == 'sigmoid'):
            self.dz = np.multiply(error,self.__sigmoid_d(self.z))
        elif(self.activation == 'tanh'):
            self.dz = np.multiply(error,self.__tanh_d(self.z))
        elif(self.activation == 'relu'):
            self.dz = np.multiply(error,self.__relu_d(self.z))
        elif(self.activation == 'softmax'):
            self.dz = error
        else:
            self.dz = 0 * error
            
        self.dw = np.dot(self.dz, self.input_data.T)
        self.db = np.sum(self.dz, axis=1, keepdims=True)
        next_error = np.dot(self.w.T, self.dz)
        return next_error
