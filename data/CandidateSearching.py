import numpy as np
import EMD as earthmover
from ot import emd
import Ochecker_reg as OC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
"""
Different attributes we need to store for a cell

Parameters:
value: (int) the value of cell in 2D segmentation numpy
highest_layer: (int) the highest layer of the cell 
lowest_layer: (int) the lowest layer of the cell
"""
class Cell:
    def __init__(self, value, highest_layer, lowest_layer):
        self.value = value
        self.highest_layer = highest_layer
        self.lowest_layer = lowest_layer

"""
Read in the file

Parameters:
file_path: (str) the path of 2D segmentation numpy file
"""
def load_array(file_path):
    try:
        return np.load(file_path)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print("An error occurred while loading the array:", e)
    return None

"""
For class Cell, fills in the info for each specific cell in 2D segmentation 
numpy file

Parameters:
array: (3D np array) result of 2D segmentation numpy file
"""
def extract_cells_info(array):
    cell_dict = {}

    if array is None:
        return cell_dict
    
    for i in range(array.shape[0]):
        layer = array[i]

        for row_idx, row in enumerate(layer):
            for col_idx, cell_value in enumerate(row):
                # Empty spot without cell
                if cell_value == 0:
                    continue
                # A new cell is detected, create a new instance for this cell
                if cell_value not in cell_dict:
                    cell_dict[cell_value] = Cell(cell_value, i, i)
                else:
                    # Update the cell info
                    cell_dict[cell_value].lowest_layer = max(cell_dict[cell_value].lowest_layer, i)
                    cell_dict[cell_value].highest_layer = min(cell_dict[cell_value].highest_layer, i)

    return cell_dict

"""
Jaccard index calculator for cell mask A and cell mask B

Parameters:
array: (3D np array) result of 2D segmentation numpy file
layerA: (int) the number of layer that cell mask A resides
layerB: (int) the number of layer that cell mask B resides
CellA_val: (int) the np value of cell A
CellB_val: (int) the np value of cell B
"""
def jaccard_index_calc(array, layerA, layerB, CellA_val, CellB_val):
    # Collect all the coordinates of Cell mask A
    cellA_coordinates = set()
    for row_idx, row in enumerate(array[layerA]):
        for col_idx, cell_value in enumerate(row):
            if cell_value == CellA_val:
                cellA_coordinates.add((row_idx, col_idx))
    
    # Collect all the coordinates of Cell mask B
    cellB_coordinates = set()
    for row_idx, row in enumerate(array[layerB]):
        for col_idx, cell_value in enumerate(row):
            if cell_value == CellB_val:
                cellB_coordinates.add((row_idx, col_idx))
    
    # Calculate the intersection and union of Cell mask A and Cell mask B
    intersection = len(cellA_coordinates.intersection(cellB_coordinates))
    union = len(cellA_coordinates.union(cellB_coordinates))
    
    if union == 0: 
        return 0.0
    jaccard_index = intersection / union
    return jaccard_index

"""
For two cells, A and B, check whether 
the lowest layer of cell A and the highest layer of cell B has a overlap from z-axis

Parameters:
array: (3D np array) result of 2D segmentation numpy file
Cell_A: (int) the np value of cell A
Cell_B: (int) the np value of cell B
"""
def overlapping_check(array, cell_A, cell_B):
    layerA = cell_A.lowest_layer
    layerB = cell_B.highest_layer
    cellA_coordinates = set()
    for row_idx, row in enumerate(array[layerA]):
        for col_idx, cell_value in enumerate(row):
            if cell_value == cell_A.value:
                cellA_coordinates.add((row_idx, col_idx))
    cellB_coordinates = set()
    for row_idx, row in enumerate(array[layerB]):
        for col_idx, cell_value in enumerate(row):
            if cell_value == cell_B.value:
                cellB_coordinates.add((row_idx, col_idx))
    union = cellA_coordinates.union(cellB_coordinates)
    layerC = cell_A.lowest_layer + 1
    maskC = array[layerC]
    for i in union:
        if maskC[i[0],i[1]] == 0:
            return True
    return False

"""
Find the candidates of pairs of cells that could have a missing mask 

Parameters:
cell_dict: (dict) A dictionary that contains three attributes for each cell
array: (3D np array) result of 2D segmentation numpy file
"""
def missing_mask_search(cell_dict, array):
    missing_cell_pairs = []
    # Detect whether there's a one-layer gap between two cells 
    for cell_A_key, cell_A in cell_dict.items():
        for cell_B_key, cell_B in cell_dict.items():
            if cell_A_key != cell_B_key:  
                if cell_B.highest_layer - cell_A.lowest_layer == 2:
                    missing_cell_pairs.append((cell_A_key, cell_B_key))
    
    missing_cell_pairs_final = []
    # Detect whether two cells are overlapped (jarccard index > 0)
    for cell_A_key, cell_B_key in missing_cell_pairs:
        cell_A = cell_dict[cell_A_key] 
        cell_B = cell_dict[cell_B_key] 
        jaccard_index = jaccard_index_calc(array, cell_A.lowest_layer, cell_B.highest_layer, cell_A.value, cell_B.value)
        if jaccard_index > 0:
            missing_cell_pairs_final.append((cell_A, cell_B))
    return  missing_cell_pairs_final

"""
Using the OT match in CellStitch to consider the situation that
not all the cells are one-to-one correspondent 

Parameters:
candidates: (2D array) storing the primary candidate for further processing
array: (3D np array) result of 2D segmentation numpy file
"""
def one_to_one_correspondence_check2(candidates, array):
    # Function to calculate the Jaccard index for a candidate pair
    def calculate_jaccard_index(pair):
        candidate_0 = pair[0]
        candidate_1 = pair[1]
        return jaccard_index_calc(
            array=array,
            layerA=candidate_0.lowest_layer,       # candidate[0]'s lowest layer
            layerB=candidate_1.highest_layer,  # candidate[1]'s highest layer
            CellA_val=candidate_0.value,          # candidate[0]'s value
            CellB_val=candidate_1.value           # candidate[1]'s value
        )
    
    # Sort the candidate pairs based on Jaccard index in descending order
    candidates_sorted = sorted(candidates, key=calculate_jaccard_index, reverse=True)
    
    # To track used candidates
    used_candidates_0 = set()
    used_candidates_1 = set()
    
    final_candidates = []

    # Greedily select pairs
    for candidate_pair in candidates_sorted:
        candidate_0 = candidate_pair[0]
        candidate_1 = candidate_pair[1]
        
        if candidate_0.value not in used_candidates_0 and candidate_1.value not in used_candidates_1:
            final_candidates.append(candidate_pair)
            used_candidates_0.add(candidate_0.value)
            used_candidates_1.add(candidate_1.value)
    
    return final_candidates


def one_to_one_correspondence_check(candidates,array):
    final_candidate = []
    for layer in range(array.shape[0]):
        lowest_layer_set = []
        highest_layer_set = []
        for candidate in candidates:
            if candidate[0].lowest_layer == layer and candidate[0] not in lowest_layer_set:
                lowest_layer_set.append(candidate[0])
            if candidate[1].highest_layer == layer + 2 and candidate[1] not in highest_layer_set:
                highest_layer_set.append(candidate[1])
        # If it's possible for a cell mask to be matched by multiple other masks
        if len(lowest_layer_set) > 1 or len(highest_layer_set) > 1:
            lowest_layer_size = []
            highest_layer_size = []
            for cell in lowest_layer_set:
                lowest_layer_size.append(np.count_nonzero(array[layer] == cell.value))
            for cell in highest_layer_set:
                highest_layer_size.append(np.count_nonzero(array[layer + 2] == cell.value))

            # Preparing for OT distribution setting
            sum_lowest_layer_size = sum(lowest_layer_size)
            sum_highest_layer_size = sum(highest_layer_size)
            normalized_lowest_layer_size = [size / sum_lowest_layer_size for size in lowest_layer_size]
            normalized_highest_layer_size = [size / sum_highest_layer_size for size in highest_layer_size]
            
            # Calculate the cost matrix for OT
            Cost_Matrix = np.zeros((len(lowest_layer_set), len(highest_layer_set)))
            for i in range(len(lowest_layer_set)):
                for j in range(len(highest_layer_set)):
                    CellA_val = lowest_layer_set[i].value
                    CellB_val = highest_layer_set[j].value
                    # 1 - Jaccard Index as the cost for transportation
                    Cost_Matrix[i, j] = 1 - jaccard_index_calc(array, layer, layer + 2, CellA_val, CellB_val)
            # Solve the OT plan
            OT_plan = emd(normalized_lowest_layer_size, normalized_highest_layer_size, Cost_Matrix)
            n, m = len(OT_plan), len(OT_plan[0])
            soft_matching = np.zeros((n, m))
            # Match the cell from n to m by finding the argmax of the OT_plan
            if n < m:
                for i in range(n):
                    soft_matching[i, OT_plan[i].argmax()] = 1
            else:
                for i in range(m):
                    soft_matching[OT_plan[i].argmax(), i] = 1
            # Reversely match the cell from m to n to find the one with least transportation cost
            '''
            for i in range(m):
                # If multiple cells in n are matched with ith cell in m
                if np.sum(soft_matching[:, i]) > 1:
                    soft_matching[np.argmax(soft_matching[:, i]), i] += 1
                    for j in range(n):
                        soft_matching[j, i] = max(0, soft_matching[j, i] - 1)
            '''
            for i in range(n):
                for j in range(m):
                    if(soft_matching[i, j] == 1):
                        final_candidate.append((lowest_layer_set[i],highest_layer_set[j]))
        elif len(lowest_layer_set) == 1 and len(highest_layer_set) == 1:
            return candidates
    return final_candidate

def print_cells_info(cell_dict):
    if not cell_dict:
        print("No cell information available.")
        return
    
    for cell_value, cell_info in cell_dict.items():
        print(f"Cell value: {cell_info.value}, Highest layer: {cell_info.highest_layer}, Lowest layer: {cell_info.lowest_layer}")


def print_candidates(candidates):
    if len(candidates) == 0:
        print("None")
    else:
        print(candidates[0].value)

"""
Calculate the earthmover's distance between two masks
Check whether the earthmover's distance across the missing mask can fit into the original trend

Parameters:
candidates: (2D array) storing the primary candidate for further processing
array: (3D np array) result of 2D segmentation numpy file
"""
def EMD_calculation(layer_A, layer_B, val_A, val_B):
    layer_A = np.where(layer_A != val_A, 0, layer_A)
    layer_A = np.where(layer_A != 0, 1, layer_A)
    layer_B = np.where(layer_B != val_B, 0, layer_B)
    layer_B = np.where(layer_B != 0, 1, layer_B)
    #If your mask is small, you can try to replace function "fast_EMD" to "EMD_2d"
    return earthmover.fast_EMD(layer_A, layer_B)
    
def EMD_processing(candidate, array, mode):
    Cell_A = candidate[0]
    Cell_B = candidate[1]
    layer_A = array[Cell_A.lowest_layer] 
    layer_B = array[Cell_B.highest_layer]
    target = EMD_calculation(layer_A, layer_B, Cell_A.value, Cell_B.value)
    Cell_A_EMD = []
    for i in range(Cell_A.highest_layer, Cell_A.lowest_layer - 1):
        layer_A = array[i] 
        layer_B = array[i + 2]
        result = EMD_calculation(layer_A, layer_B, Cell_A.value, Cell_A.value)
        if result != 0:
            Cell_A_EMD.append(result)
    
    Cell_B_EMD = []
    for i in range(Cell_B.highest_layer, Cell_B.lowest_layer - 1):
        layer_A = array[i] 
        layer_B = array[i + 2]
        result = EMD_calculation(layer_A, layer_B, Cell_B.value, Cell_B.value)
        if result != 0:
            Cell_B_EMD.append(result)
    
    features = []

    try:
        q1_x1 = np.percentile(Cell_A_EMD, 25)
        q2_x1 = np.percentile(Cell_A_EMD, 50)
        q3_x1 = np.percentile(Cell_A_EMD, 75)
        min_x1 = min(Cell_A_EMD)
        max_x1 = max(Cell_A_EMD)

        q1_x2 = np.percentile(Cell_B_EMD, 25)
        q2_x2 = np.percentile(Cell_B_EMD, 50)
        q3_x2 = np.percentile(Cell_B_EMD, 75)
        min_x2 = min(Cell_B_EMD)
        max_x2 = max(Cell_B_EMD)

        if mode == 'minmax':
            features = [q1_x1, q2_x1, q3_x1, min_x1, max_x1, q1_x2, q2_x2, q3_x2, min_x2, max_x2, target]
        else: 
            features = [q1_x1, q2_x1, q3_x1, q1_x1, q3_x1, q1_x2, q2_x2, q3_x2, q1_x2, q3_x2, target]

        return features
    
    except:
        return -1
        

def EMD_trainning(cell, array):
    Cell_EMD = []
    for i in range(cell.highest_layer, cell.lowest_layer - 1):
        try:
            layer_A = array[i] 
            layer_B = array[i + 2]
            result = EMD_calculation(layer_A, layer_B, cell.value, cell.value)
        except:
            return []
        if result != 0:
            Cell_EMD.append(result)
    return Cell_EMD


"""
OChker function, checking whether the surface area changing of the cell is 
approximately linear or quadratic 

Parameters:
CellA_val: (int) the np value of cell A
CellB_val: (int) the np value of cell B
CellA_highest: (int) the highest mask of cell A
CellA_lowest: (int) the lowest mask of cell A
CellB_highest: (int) the highest mask of cell B
CellB_lowest: (int) the lowest mask of cell B
arraynew: (3D np array) result of 2D segmentation numpy file
"""
def OChker(cellA_val, cellB_val, cellA_highest, cellA_lowest, cellB_highest, cellB_lowest, arraynew, sigma):
    for layer in range(cellB_highest, cellB_lowest + 1):
        for i in range(len(arraynew[layer])):
            for j in range(len(arraynew[layer, 0])):
                if(arraynew[layer,i, j] == cellB_val):
                    arraynew[layer,i, j] = cellA_val
    arraynew = np.delete(arraynew, cellA_lowest + 1, axis=0)
    result = OC.O_index_visualizor(cellA_val, cellA_highest, cellB_lowest - 1, arraynew, sigma)
    if(result == 1):
        return True
    else:
        return False

def new_Och(cellA_val, cellB_val, cellA_highest, cellA_lowest, cellB_highest, cellB_lowest, arraynew):
    for layer in range(cellB_highest, cellB_lowest + 1):
        for i in range(len(arraynew[layer])):
            for j in range(len(arraynew[layer, 0])):
                if(arraynew[layer,i, j] == cellB_val):
                    arraynew[layer,i, j] = cellA_val
    arraynew = np.delete(arraynew, cellA_lowest + 1, axis=0)
    result = OC.O_index_result(cellA_val, cellA_highest, cellB_lowest - 1, arraynew)
    return result

def main(name, tolerance, mode):
    file_path = name

    if file_path.endswith('.npy'):
        array = np.load(file_path)
    elif file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            array = f['cellstitch_masks'][:]

    # Cell info extraction
    cell_dict = extract_cells_info(array)
    primary_candidates = missing_mask_search(cell_dict, array)
    final_candidates = one_to_one_correspondence_check2(primary_candidates,array)

    results = []

    for candidate in final_candidates:
        arraynew = array.copy() 

        # Processing Cell's Geometric Info (2D)
        EMDs = EMD_processing(candidate, arraynew, mode)
        if EMDs == -1:
            continue
        
        # Processing Cell's Topological Info (3D)
        Oix = new_Och(candidate[0].value, candidate[1].value, candidate[0].highest_layer, candidate[0].lowest_layer, candidate[1].highest_layer, candidate[1].lowest_layer, arraynew)

        # Statistical parameters learned from the pretrained model
        li_m, li_std = 1, 0
        qu_m, qu_std = 0.9596848301316223, 0.12202991544763167
        Oix_zs = 0

        if Oix[1] == 0:
            if li_m == 0:
                Oix_zs = 0
            else:
                Oix_zs = Oix[0] - li_m    
        else:
            Oix_zs = (Oix[0] - qu_m) / qu_std
        EMDs.append(Oix_zs - tolerance)

        # Concatenating all the info as the input for the network
        # Dim_input = 12 
        input_dim = 12  
        model = MLP(input_dim=input_dim)
        model.load_state_dict(torch.load('mlp_model_epoch_50.pth'))
        model.eval()
        inputs_tensor = torch.tensor(EMDs, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(inputs_tensor)
        prediction = outputs.round().numpy()

        if prediction == 1:
            results.append([candidate[0].value, candidate[1].value, candidate[0].lowest_layer + 1])
    return results

if __name__ == "__main__":
    main()