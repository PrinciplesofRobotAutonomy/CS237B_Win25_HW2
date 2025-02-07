import torch
import torch.nn as nn
import torch.nn.functional as F

DIM_IMG = (224, 224)

class AccelerationLaw(nn.Module):
    """
    PyTorch layer to evaluate the acceleration law:

        a = g * (sin(th) - mu * cos(th))

    g is a trainable parameter because the units of acceleration in the
    dataset are pixels/frame^2, and the conversion from 9.81 m/s^2 to these
    units are unknown.
    """

    def __init__(self):
        super(AccelerationLaw, self).__init__()

        ################### Your code starts here ########################
        # TODO: Define a trainable Parameter for scalar g: acceleration due to gravity
        # Use nn.Parameter with initial value 16


        ################### Your code starts here ########################

    def forward(self, mu, th):

        # Reshape to prevent bad broadcasting
        mu = mu.reshape(-1, 1)
        th = th.reshape(-1, 1)

        ################### Your code starts here ########################
        # Use the acceleration law to compute a and return it
        
        pass # REMOVE THIS LINE
    
        ################### Your code starts here ########################

class AccelerationPredictionNetwork(nn.Module):
    def __init__(self):
        super(AccelerationPredictionNetwork, self).__init__()

        ################### Your code starts here ########################
        # Create a prediction network design of your choice here.
        # We recommend using Conv2D, pooling and dropout layers,
        # followed by specific layers to represent your final p_class and mu



        self.p_class = ...
        self.mu = ...

        ################### Your code ends here ########################

        self.acceleration_law = AccelerationLaw()

    def get_p_class_output(self, input):
        """
        function to output and store the p_class output.

        Parameter:
            input: the model img input

        Return:
            returns the output from the p_class layer after performing Softmax 
        """
        ################### Your code starts here ########################


        p_class_output = ...
    
        ################### Your code ends here ########################
        return p_class_output

    def forward(self, img, th):

        ################### Your code starts here ########################
        # Pass the inputs through the layers to compute p_class and then mu

        mu = ...
        ################### Your code ends here ########################
        
        a_pred = self.acceleration_law(mu, th)
        return a_pred

class BaselineNetwork(nn.Module):
    def __init__(self):
        super(BaselineNetwork, self).__init__()

        ################### Your code starts here ########################
        # Copy the model layers you used in your design for the AccelerationPredictionNetwork

        ################### Your code ends here ########################

    def forward(self, img):

        ################### Your code starts here ########################
        # Compute a_pred similar to before except without the acceleration law




        a_pred = ...

        ################### Your code ends here ########################
        return a_pred

def loss(a_actual, a_pred):
    """
    Loss function: L2 norm of the error between a_actual and a_pred for a batch of samples.
    """
    ################### Your code starts here ########################
    # Compute the MSE loss between the actual and predicted accelerations

    loss = None # Replace this line

    ################### Your code ends here ########################

    return loss
