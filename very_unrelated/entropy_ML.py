import numpy as np

# Parameters
Eyes = ["Blue", "Green", "Brown", "Black"]
N_fingers = np.arange(8)
Sex = ["Male", "Female"]
N_ppl = 100
N_killers = 2

Columns = 4
ColNames = ["Eyes","N_Fingers","Sex","Killer"]

# Set random seed for reproducibility
np.random.seed(42)

# Define probabilities for each parameter
eye_probs = [0.25, 0.25, 0.25, 0.25]  # Blue, Green, Brown, Black
finger_probs = np.ones_like(N_fingers)/8   # 3, 4, 5, 6 fingers
sex_probs = [0.5, 0.5]  # Male, Female

# Generate dataset
generated_eyes = np.random.choice(Eyes, size=N_ppl, p=eye_probs)
generated_fingers = np.random.choice(N_fingers, size=N_ppl, p=finger_probs)
generated_sex = np.random.choice(Sex, size=N_ppl, p=sex_probs)

generated_killer_num = np.zeros_like(generated_eyes)
generated_killer_num[0:N_killers] = 1
generated_killer = np.empty_like(generated_killer_num)
generated_killer[generated_killer_num == 1] = "Killer"
generated_killer[generated_killer_num == 0] = "Civil"

# Optionally combine into a dataset
people = np.column_stack([generated_eyes, generated_fingers, generated_sex, generated_killer_num])

# Example: print first 5 rows
print("First 5 rows of generated dataset:")
print(people[:5])



def countOf(data, column_i,value):
    eye_unique, eye_counts = np.unique(data[:, column_i], return_counts=True)   
    return  eye_counts[eye_unique == value]




def calcEntropy(data,column_i):
    """
    Calculates the standard marginal entropy H(X) of a single column.
    """
    vals, counts = np.unique(data[:, column_i], return_counts=True)  
    probs = counts / np.sum(counts)
    
    # Using element-wise numpy operations instead of a loop
    entropy = -np.sum(probs * np.log2(probs)) 
    return entropy

def calcJoint_Entropy(data, columns_list):
    """
    Calculates the true Joint Entropy H(X, Y, ...) for a selection of columns.
    Matches unique combinations of rows.
    """
    # Extract only the columns we want to find the joint entropy for
    sub_data = data[:, columns_list]
    
    # axis=0 finds unique combined row patterns
    unique_rows, counts = np.unique(sub_data, axis=0, return_counts=True)
    probs = counts / np.sum(counts)
    
    joint_entropy = -np.sum(probs * np.log2(probs))
    return joint_entropy

def calcConditional_Entropy(data, target_col, attr_col):
    """
    Calculates Conditional Entropy H(Y | A)
    How much uncertainty remains in target_col (Y) given that we know attr_col (A).
    """
    # 1. Get the unique values 'v' and counts '|S_v|' of our attribute column (A)
    vals_A, counts_A = np.unique(data[:, attr_col], return_counts=True)
    total_rows = data.shape[0] # |S|
    
    cond_entropy = 0.0
    
    # 2. Loop through each unique value v of attribute A
    for v, count_v in zip(vals_A, counts_A):
        # S_v: Get the subset of rows where the attribute equals v
        subset = data[data[:, attr_col] == v]
        
        # H(Y_v): Calculate the normal entropy of the target column inside this subset
        entropy_of_subset = calcEntropy(subset, target_col)
        
        # Weight (|S_v| / |S|)
        weight = count_v / total_rows
        
        # Accumulate weighted entropy
        cond_entropy += weight * entropy_of_subset
        
    return cond_entropy


print("========= Stats ============")
print(f"Dataset Joint entropy: {calcJoint_Entropy(people,np.arange(Columns) ) :.3f}")
print(f"Dataset Joint entropy - w/o killers: {calcJoint_Entropy(people[:,:3],np.arange(Columns-1)) :.3f}")

for col_i in range(Columns):
    
    H = calcEntropy(people,col_i)
    print(f"{ColNames[col_i]} : Entropy {H :.3f} " )


for col_i in range(Columns-1):
    H_YA = calcConditional_Entropy(people, target_col=3,attr_col= col_i)
    print(f"{ColNames[col_i]}  Cond. Entropy H(Y|A): {H_YA :.3f} " )  



for col_i in range(Columns-1):
    H_YA = calcConditional_Entropy(people, target_col=3,attr_col= col_i)
    H = calcEntropy(people,3)
    print(f"{ColNames[col_i]}  inf. gain I(Y,A): {H - H_YA :.3f} " )  



