import numpy as np
import random
import math

# Import maxflow (PyMaxflow library) which can be used to solve min-cut problem
import maxflow

# Set seeds for random generators to get reproducible results
random.seed(0)
np.random.seed(0)

def perform_min_cut(unary_potential_foreground, unary_potential_background, pairwise_potential):
    """
    We provide a simple fuction to perform min cut using PyMaxFlow library. You
    may use this function to implement your algorithm if you wish. Feel free to
    modify this function as desired, or implement your own function to perform
    min cut.  

    args:
        unary_potential_foreground - A single channel NumPy array specifying the
            source (foreground) unary potentials for each pixel in the image
        unary_potential_background - A single channel NumPy array specifying the
            sink (background) unary potentials for each pixel in the image
        pairwise_potential - A single channel NumPy array specifying the pairwise
            potentials. We assume a graph where each pixel in the image is 
            connected to its four neighbors (left, right, top, and bottom). 
            Furthermore, we assume that the pairwise potential for all these 4
            edges are same, and set to the value of pairwise_potential at that 
            pixel location
    """    
    
    # create graph
    maxflow_graph = maxflow.Graph[float]()
    
    # add a node for each pixel in the image
    nodeids = maxflow_graph.add_grid_nodes(unary_potential_foreground.shape[:2])

    # Add edges for pairwise potentials. We use 4 connectivety, i.e. each pixel 
    # is connected to its 4 neighbors (up, down, left, right). Also we assume
    # that pairwise potential for all these 4 edges are same
    # Feel free to change this if you wish
    maxflow_graph.add_grid_edges(nodeids, pairwise_potential)

    # Add unary potentials
    maxflow_graph.add_grid_tedges(nodeids, unary_potential_foreground, unary_potential_background)

    maxflow_graph.maxflow()
    
    # Get the segments of the nodes in the grid.
    mask_bg = maxflow_graph.get_grid_segments(nodeids)
    mask_fg = (1 - mask_bg.astype(np.uint8))* 255

    return mask_fg

def get_intensity(r,g,b):
    return (0.3333*(r) + 0.3333*(g) + 0.3334*(b))

class ImageSegmenter:
    def __init__(self):
        pass
    
    def segment_image(self, im_rgb, im_aux, im_box):
        # TODO: Modify this function to implement your algorithm
        # Use im_aux only      
        
        # Intensity distribution of aux image
        im_aux_it = get_intensity(im_aux[:,:,0], im_aux[:,:,1], im_aux[:,:,2])          
        im_aux_it_in = im_aux_it[np.nonzero(im_box)] 
        im_aux_it_out = np.delete(np.ndarray.flatten(im_aux_it), np.nonzero(np.ndarray.flatten(im_box)))
        
        # foreground
        It = np.mean(im_aux_it_in)  
        # background 
        Is = np.mean(im_aux_it_out)  
   
        # Perform simple min cut
        sigma = 0.1  
        # Foreground potential
        unary_potential_foreground = np.exp(-(np.absolute(im_aux_it - It)**2)/2*(sigma**2)) 

        # Background potential 
        unary_potential_background = np.exp(-(np.absolute(im_aux_it - Is)**2)/2*(sigma**2))       
            
        # Pairwise potential 
        Ipq = np.diff(im_aux_it)
        pairwise_potential = np.exp(-np.absolute(Ipq)/(2*sigma**2))
        # reshape to n*n matrix
        pairwise_potential = np.hstack([pairwise_potential, np.zeros((len(pairwise_potential[:,0]),1))])

        # Perfirm min cut to get segmentation mask
        im_mask = perform_min_cut(unary_potential_foreground, unary_potential_background, 
                                  pairwise_potential)
                                    
        # Get rid of any "foreground" pixels outside of box
        im_mask = np.multiply(im_mask, im_box.astype(np.float32)/255)
        
        return im_mask
